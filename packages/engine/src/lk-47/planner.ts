import type { BattleState, TacticType } from '../battlecast/engine/combat.js';
import { creatureDistance } from '../battlecast/engine/combat-geometry.js';
import { withBattlecastRngAsync } from '../battlecast/engine/dice.js';
import type { Creature } from '../battlecast/types/monster.js';
import { moveTowardActionId, type LegalAction, type LegalActionCatalogue } from '../legal-actions.js';
import { createRng, normalizeSeed, type SeededRng } from '../random.js';
import { copyBattle } from './battle-copy.js';
import { chooseBotReaction, chooseBotSmite } from './bot-defaults.js';
import type { Lk47DecisionContext, Lk47Policy, Lk47Simulator } from './variants.js';

export interface PlannerConfig {
  // How every simulated creature plays after the current one's turn: a Battlecast bot tactic, or LK-47's own greedy policy.
  model: TacticType | 'greedy';
  // Successive halving: give each surviving candidate this many samples in total, then keep the best `keep`.
  schedule: Array<{ samples: number; keep: number }>;
}

type MoveTo = Extract<LegalAction, { type: 'move_to' }>;

const maxTurnSteps = 12;
const decidedOutcome = 100;

export function createPlannerPolicy(config: PlannerConfig): Lk47Policy {
  return {
    async chooseAction(context) {
      const { state, activeCreature, catalogue, turn } = context;
      const prompt = chooseBotReaction(activeCreature, catalogue) ?? chooseBotSmite(catalogue);
      if (prompt) return prompt;
      const candidates = selectCandidates(state, activeCreature, catalogue);
      if (!turn || candidates.length === 1) return candidates[0];

      const seed = decisionSeed(state, activeCreature, catalogue);
      let pool = candidates.map((action) => ({ action, total: 0, samples: 0 }));
      for (const { samples, keep } of config.schedule) {
        for (const entry of pool) {
          // Sample k uses the same dice for every candidate, so candidates are compared on equal luck.
          for (; entry.samples < samples; entry.samples += 1) {
            entry.total += await rollout(context, entry.action, seed + entry.samples, config.model);
          }
        }
        pool = pool
          .sort((left, right) => right.total / right.samples - left.total / left.samples)
          .slice(0, keep);
      }
      return pool[0].action;
    },
  };
}

// Every non-movement option, plus the few squares worth moving to: next to each enemy, next to each dying ally,
// and the one farthest from the nearest enemy. Wild Shape forms and Magic Missile splits are thinned out.
function selectCandidates(state: BattleState, active: Creature, catalogue: LegalActionCatalogue): LegalAction[] {
  const moves = catalogue.actions.filter((action): action is MoveTo => action.type === 'move_to');
  const forms = catalogue.actions
    .filter((action) => action.type === 'class_feature' && action.feature === 'wild_shape')
    .sort((left, right) =>
      (right.type === 'class_feature' ? right.beastTempHp ?? 0 : 0) - (left.type === 'class_feature' ? left.beastTempHp ?? 0 : 0)
    )
    .slice(0, 3);
  const kept = catalogue.actions.filter((action) =>
    action.type !== 'move_to' &&
    !(action.type === 'class_feature' && action.feature === 'wild_shape') &&
    !(action.type === 'spell' && action.effectKind === 'auto_darts' && new Set(action.targetIds).size > 1)
  );
  const anchors = state.creatures
    .filter((creature) => creature.isAlive && creature.id !== active.id && (creature.team !== active.team ? !creature.dying : creature.dying))
    .map((creature) => creature.position);
  const squares = new Set<MoveTo>();
  for (const anchor of anchors) {
    const nearest = closest(moves, (move) => gridDistance(move.destination, anchor));
    if (nearest) squares.add(nearest);
  }
  const safest = closest(moves, (move) => -(move.nearestEnemyDistanceAfterFt ?? 0));
  if (safest) squares.add(safest);
  return [...kept, ...forms, ...squares];
}

async function rollout(context: Lk47DecisionContext, action: LegalAction, seed: number, model: PlannerConfig['model']): Promise<number> {
  const { state, activeCreature, turn, simulator } = context;
  if (!turn) throw new Error('a rollout needs the turn state');
  const copy = copyBattle(state);
  const active = copy.creatures.find((creature) => creature.id === activeCreature.id);
  if (!active) throw new Error(`no creature ${activeCreature.id} in the battle`);
  const ownTurn = structuredClone(turn);
  const rng = createRng(seed);
  await withBattlecastRngAsync(rng, async () => {
    let ended = await simulator.applyAction(copy, active, action, ownTurn, rng);
    for (let step = 1; !ended && step < maxTurnSteps && !copy.isComplete && active.isAlive; step += 1) {
      ended = await simulator.applyAction(copy, active, greedyStep(copy, simulator.actionsInPlace(copy, active, ownTurn)), ownTurn, rng);
    }
    simulator.finishTurn(copy, active);
    if (model !== 'greedy') copy.teamTactics = { ...copy.teamTactics, red: model, blue: model };
    // One round: everyone else acts once, up to this creature's next turn.
    const order = copy.initiativeOrder;
    for (let offset = 1; offset < order.length && !copy.isComplete; offset += 1) {
      const index = (copy.turnIndex + 1) % order.length;
      if (index === 0) copy.round += 1;
      if (copy.matchMaxRounds !== undefined && copy.round > copy.matchMaxRounds) break;
      copy.turnIndex = index;
      const creature = copy.creatures.find((candidate) => candidate.id === order[index]);
      if (!creature?.isAlive) continue;
      if (model === 'greedy') await playGreedyTurn(copy, creature, simulator, rng);
      else simulator.playScriptedTurn(copy, creature, model);
    }
  });
  return evaluate(copy, active.team);
}

// The best immediate gain from where the creature stands; if there is none and no enemy is within 5 ft,
// one move toward the nearest enemy through the menu's own move_toward, then look again.
async function playGreedyTurn(state: BattleState, active: Creature, simulator: Lk47Simulator, rng: SeededRng): Promise<void> {
  const turn = simulator.beginTurn(state, active);
  if (!turn) return;
  let moved = false;
  let ended = false;
  for (let step = 0; !ended && step < maxTurnSteps && !state.isComplete && active.isAlive; step += 1) {
    let action = greedyStep(state, simulator.actionsInPlace(state, active, turn));
    if (action.type === 'end_turn' && !moved) {
      moved = true;
      const enemies = state.creatures.filter((creature) => creature.team !== active.team && creature.isAlive && !creature.dying);
      const target = closest(enemies, (enemy) => creatureDistance(active, enemy));
      if (target && creatureDistance(active, target) > 5) {
        action = { id: moveTowardActionId(target.id), type: 'move_toward', targetId: target.id, targetName: target.displayName };
      }
    }
    ended = await simulator.applyAction(state, active, action, turn, rng);
  }
  simulator.finishTurn(state, active);
}

function greedyStep(state: BattleState, catalogue: LegalActionCatalogue): LegalAction {
  const smite = chooseBotSmite(catalogue);
  if (smite) return smite;
  let best: LegalAction | undefined;
  let bestValue = 0;
  for (const action of catalogue.actions) {
    const value = immediateValue(state, action);
    if (value > bestValue) {
      best = action;
      bestValue = value;
    }
  }
  return best ?? catalogue.actions.find((action) => action.type === 'end_turn') ?? catalogue.actions[0];
}

function immediateValue(state: BattleState, action: LegalAction): number {
  if (action.type === 'stabilise') return 15;
  if (action.type === 'spell' && action.expectedHealing && action.targetId) {
    const ally = state.creatures.find((creature) => creature.id === action.targetId);
    if (!ally) return 0;
    if (ally.dying) return action.expectedHealing + 20;
    return ally.currentHp < ally.maxHp / 2 ? action.expectedHealing : 0;
  }
  const damage = action.type === 'attack' || action.type === 'linked_bonus_damage' || action.type === 'spell' || action.type === 'class_feature'
    ? action.expectedDamage ?? 0
    : 0;
  const targetId = 'targetId' in action ? action.targetId : undefined;
  const target = state.creatures.find((creature) => creature.id === targetId);
  return target && damage > 0 && damage >= target.currentHp ? damage + 10 : damage;
}

function evaluate(state: BattleState, team: Creature['team']): number {
  if (state.isComplete) {
    if (state.winner === team) return decidedOutcome;
    return state.winner === 'draw' ? 0 : -decidedOutcome;
  }
  let score = 0;
  for (const creature of state.creatures) score += (creature.team === team ? 1 : -1) * worth(creature);
  return score;
}

// A standing creature is worth 1 plus its HP fraction, less if it is disabled; a dying one can still be healed back.
function worth(creature: Creature): number {
  if (!creature.isAlive) return 0;
  if (creature.dying) return 0.3;
  const disabled = creature.conditions.some((condition) =>
    condition === 'paralyzed' || condition === 'stunned' || condition === 'incapacitated' || condition === 'unconscious' || condition === 'restrained'
  );
  return (disabled ? 0.6 : 1) + Math.max(0, creature.currentHp) / creature.maxHp;
}

function decisionSeed(state: BattleState, active: Creature, catalogue: LegalActionCatalogue): number {
  const board = state.creatures.map((creature) => `${creature.id}:${creature.currentHp}:${creature.position.x},${creature.position.y}`);
  return normalizeSeed(`${state.round}|${state.turnIndex}|${active.id}|${catalogue.actions.length}|${board.join('|')}`);
}

function closest<T>(items: T[], distance: (item: T) => number): T | undefined {
  return items.reduce<T | undefined>((best, item) => (best === undefined || distance(item) < distance(best) ? item : best), undefined);
}

function gridDistance(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
}
