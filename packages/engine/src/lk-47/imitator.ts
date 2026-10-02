import { executeTurn } from '../battlecast/engine/ai-turn.js';
import { selectTarget } from '../battlecast/engine/ai-targeting.js';
import type { BattleLog, BattleState, TacticType } from '../battlecast/engine/combat.js';
import { withBattlecastRng } from '../battlecast/engine/dice.js';
import type { Creature } from '../battlecast/types/monster.js';
import type { LegalAction, LegalActionCatalogue } from '../legal-actions.js';
import { createRng, normalizeSeed } from '../random.js';
import { copyBattle } from './battle-copy.js';
import { chooseBotReaction, chooseBotSmite } from './bot-defaults.js';
import type { Lk47Policy } from './variants.js';

type Cell = { x: number; y: number };

export type PlannedStep =
  | { kind: 'move'; to: Cell }
  | { kind: 'attack'; actionName: string; targetId: string }
  | { kind: 'area'; actionName: string; center: Cell; direction?: Cell }
  | { kind: 'heal'; actionName: string; targetIds: string[] }
  | { kind: 'stabilise'; targetId: string }
  | { kind: 'wild-shape'; beastName: string }
  | { kind: 'named'; actionName: string; targetName?: string };

// Runs the scripted bot's turn on a private copy under private dice, after the match has processed the turn start.
export function planBotTurn(state: BattleState, creatureId: string, tactic: TacticType, seed: number): PlannedStep[] {
  const copy = copyBattle(state);
  const creature = copy.creatures.find((candidate) => candidate.id === creatureId);
  if (!creature) throw new Error(`no creature ${creatureId} in the battle`);
  copy.teamTactics = { ...copy.teamTactics, [creature.team]: tactic };
  withBattlecastRng(createRng(seed), () => executeTurn(copy, creature, { turnStartAlreadyProcessed: true }));
  return readSteps(copy, creature);
}

export function createImitatorPolicy(tactic: TacticType): Lk47Policy {
  const turns = new WeakMap<BattleState, { key: string; steps: PlannedStep[]; next: number }>();
  return {
    async chooseAction({ state, activeCreature, catalogue }) {
      const prompt = chooseBotReaction(activeCreature, catalogue) ?? chooseBotSmite(catalogue);
      if (prompt) return prompt;
      const key = `${state.round}:${state.turnIndex}:${activeCreature.id}`;
      let turn = turns.get(state);
      if (turn?.key !== key) {
        turn = { key, steps: planBotTurn(state, activeCreature.id, tactic, planSeed(state, activeCreature)), next: 0 };
        turns.set(state, turn);
      }
      while (turn.next < turn.steps.length) {
        const step = turn.steps[turn.next];
        turn.next += 1;
        const action = resolveStep(step, state, activeCreature, catalogue, tactic);
        if (!action) continue;
        // One spell menu action covers every ray or beam the bot rolled for it.
        if (action.type === 'spell') {
          while (turn.next < turn.steps.length && sameAttack(turn.steps[turn.next], action.actionName)) turn.next += 1;
        }
        return action;
      }
      return catalogue.actions.find((action) => action.type === 'end_turn') ?? catalogue.actions[0];
    },
  };
}

function planSeed(state: BattleState, active: Creature): number {
  const board = state.creatures.map((creature) => `${creature.id}:${creature.currentHp}:${creature.position.x},${creature.position.y}`);
  return normalizeSeed(`${state.round}|${state.turnIndex}|${active.id}|${board.join('|')}`);
}

// Events carry creature ids and come first; logs fill in actions that leave no event (buffs, class features, Dash).
// Display names repeat across teams, so a log is only trusted for actions of the acting creature during its own turn.
function readSteps(copy: BattleState, self: Creature): PlannedStep[] {
  const steps: PlannedStep[] = [];
  const healNameAt = new Map<number, string>();
  const logsAt = new Map<number, BattleLog[]>();
  for (const log of copy.logs) logsAt.set(log.eventIndex, [...(logsAt.get(log.eventIndex) ?? []), log]);

  for (let index = 0; index <= copy.events.length; index += 1) {
    for (const log of logsAt.get(index) ?? []) {
      if (log.actor !== self.displayName) continue;
      if (log.type === 'heal') {
        healNameAt.set(log.eventIndex, log.action);
      } else if (log.action === 'Dash' || log.action === 'Cunning Action: Disengage' || log.action === 'Nimble Escape: Disengage') {
        const named: PlannedStep = { kind: 'named', actionName: log.action === 'Dash' ? 'Dash' : 'Bonus Disengage' };
        const lastMove = steps.findLastIndex((step) => step.kind === 'move');
        steps.splice(lastMove >= 0 ? lastMove : steps.length, 0, named);
      } else if (log.type === 'special' || log.action === 'Disengage') {
        steps.push({ kind: 'named', actionName: log.action, targetName: targetNameIn(log.details) });
      }
    }
    if (index === copy.events.length) break;
    const event = copy.events[index];
    const last = steps.at(-1);
    if (event.kind === 'move' && event.creatureId === self.id) {
      if (last?.kind === 'move') last.to = { ...event.to };
      else steps.push({ kind: 'move', to: { ...event.to } });
    } else if (event.kind === 'attack' && event.attackerId === self.id && event.cause !== 'opportunity') {
      steps.push({ kind: 'attack', actionName: event.actionName, targetId: event.targetId });
    } else if (event.kind === 'aoe' && event.attackerId === self.id) {
      steps.push({ kind: 'area', actionName: event.spellName ?? '', center: { ...event.center }, direction: event.direction && { ...event.direction } });
    } else if (event.kind === 'stabiliseAlly' && event.actorId === self.id) {
      steps.push({ kind: 'stabilise', targetId: event.creatureId });
    } else if (event.kind === 'wildShape' && event.creatureId === self.id && event.beastName) {
      steps.push({ kind: 'wild-shape', beastName: event.beastName });
    } else if (event.kind === 'heal' && healNameAt.has(index)) {
      const actionName = healNameAt.get(index) ?? '';
      if (last?.kind === 'heal' && last.actionName === actionName) last.targetIds.push(event.creatureId);
      else steps.push({ kind: 'heal', actionName, targetIds: [event.creatureId] });
    }
  }
  // A cast is logged and then resolved as an event; keep only the event, which names the target by id.
  return steps.filter((step, index) => step.kind !== 'named' || stepName(steps[index + 1]) !== step.actionName);
}

function stepName(step: PlannedStep | undefined): string | undefined {
  if (!step) return undefined;
  if (step.kind === 'stabilise') return 'Stabilise';
  if (step.kind === 'wild-shape') return 'Wild Shape';
  return step.kind === 'move' ? undefined : step.actionName;
}

function targetNameIn(details: string): string | undefined {
  return details.match(/ (?:on|to) (.+)\.$/)?.[1];
}

function sameAttack(step: PlannedStep, actionName: string): boolean {
  return step.kind === 'attack' && step.actionName === actionName;
}

function resolveStep(
  step: PlannedStep,
  state: BattleState,
  self: Creature,
  catalogue: LegalActionCatalogue,
  tactic: TacticType,
): LegalAction | undefined {
  const actions = catalogue.actions;
  switch (step.kind) {
    case 'move': {
      const moves = actions.filter((action): action is Extract<LegalAction, { type: 'move_to' }> => action.type === 'move_to');
      const best = moves.reduce<Extract<LegalAction, { type: 'move_to' }> | undefined>((chosen, action) =>
        !chosen ||
        gridDistance(action.destination, step.to) < gridDistance(chosen.destination, step.to) ||
        (gridDistance(action.destination, step.to) === gridDistance(chosen.destination, step.to) && action.distanceFt < chosen.distanceFt)
          ? action
          : chosen, undefined);
      return best && gridDistance(best.destination, step.to) < gridDistance(self.position, step.to) ? best : undefined;
    }
    case 'attack': {
      const usable = actions.filter((action) =>
        (action.type === 'attack' || action.type === 'spell') && action.actionName === step.actionName
      );
      const exact = usable.find((action) => targetsOf(action).includes(step.targetId));
      if (exact) return exact;
      const retarget = selectTarget(state, self, targetStrategyOf(tactic));
      return retarget ? usable.find((action) => targetsOf(action).includes(retarget.id)) : undefined;
    }
    case 'area': {
      const spells = actions.filter((action): action is Extract<LegalAction, { type: 'spell' }> =>
        action.type === 'spell' && action.actionName === step.actionName
      );
      return spells.reduce<Extract<LegalAction, { type: 'spell' }> | undefined>((chosen, action) =>
        !chosen || areaMiss(action, step, state) < areaMiss(chosen, step, state) ? action : chosen, undefined);
    }
    case 'heal':
      return actions.find((action) =>
        action.type === 'spell' && action.actionName === step.actionName &&
        (targetsOf(action).length === 0 || targetsOf(action).some((id) => step.targetIds.includes(id)))
      );
    case 'stabilise':
      return actions.find((action) => action.type === 'stabilise' && action.targetId === step.targetId);
    case 'wild-shape':
      return actions.find((action) => action.type === 'class_feature' && action.feature === 'wild_shape' && action.beastName === step.beastName);
    case 'named':
      return resolveNamed(step, self, actions);
  }
}

function resolveNamed(step: Extract<PlannedStep, { kind: 'named' }>, self: Creature, actions: LegalAction[]): LegalAction | undefined {
  if (step.actionName === 'Dash') return actions.find((action) => action.type === 'dash');
  if (step.actionName === 'Disengage') return actions.find((action) => action.id === 'disengage');
  if (step.actionName === 'Bonus Disengage') return actions.find((action) => action.id === 'bonus_disengage');
  const feature = actions.find((action) => action.type === 'class_feature' && action.label === step.actionName);
  if (feature) return feature;
  const named = actions.filter((action) =>
    (action.type === 'spell' && action.actionName === step.actionName) ||
    (action.type === 'spell_retarget' && action.spellName === step.actionName)
  );
  const wanted = step.targetName === 'self' ? self.displayName : step.targetName;
  return named.find((action) => wanted !== undefined && targetNameOf(action) === wanted) ?? named[0];
}

function targetsOf(action: LegalAction): string[] {
  if (action.type !== 'attack' && action.type !== 'spell') return [];
  if (action.type === 'spell' && action.targetIds) return action.targetIds;
  return action.targetId ? [action.targetId] : [];
}

function targetNameOf(action: LegalAction): string | undefined {
  return action.type === 'spell' || action.type === 'spell_retarget' ? action.targetName : undefined;
}

function areaMiss(action: Extract<LegalAction, { type: 'spell' }>, step: Extract<PlannedStep, { kind: 'area' }>, state: BattleState): number {
  if (action.center) return gridDistance(action.center, step.center);
  if (action.direction && step.direction) return gridDistance(action.direction, step.direction);
  const target = state.creatures.find((creature) => creature.id === action.targetId);
  return target ? gridDistance(target.position, step.center) : Number.MAX_SAFE_INTEGER;
}

function gridDistance(a: Cell, b: Cell): number {
  return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
}

function targetStrategyOf(tactic: TacticType): 'nearest' | 'weakest' | 'smart' {
  if (tactic === 'kiting') return 'weakest';
  return tactic === 'smart' ? 'smart' : 'nearest';
}
