import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  estimateAttackRollBudget,
  generateLegalActions,
  isAgentId,
  lk47Simulator,
  runAgentMatch,
  runAgentMatchAsync,
  statusPressureHeroMirrorScenario,
  type D20benchScenario,
  type LegalAction,
  type ReplayEvent,
} from '../src/index.js';
import { buildHero } from '../src/battlecast/data/heroes.js';
import { maps } from '../src/battlecast/data/maps.js';
import { beginBattlecastControlledTurn } from '../src/battlecast/engine/ai-turn.js';
import { initBattle, type BattleState } from '../src/battlecast/engine/combat.js';
import { withBattlecastRng, withBattlecastRngAsync } from '../src/battlecast/engine/dice.js';
import type { Creature } from '../src/battlecast/types/monster.js';
import { buildMovementBlockedSet, buildSightBlockedSet } from '../src/battlecast/types/terrain.js';
import { createBattlecastCreatures } from '../src/battlecast-runner.js';
import { planBotTurn } from '../src/lk-47/imitator.js';
import { lk47Policies } from '../src/lk-47/variants.js';
import { createRng } from '../src/random.js';

describe('LK-47 harness wiring', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('accepts only registered LK-47 ids', () => {
    expect(isAgentId('lk-47')).toBe(true);
    expect(isAgentId('lk-47.pass')).toBe(true);
    expect(isAgentId('lk-47.unknown')).toBe(false);
  });

  it('refuses to play outside the actual-action protocol', async () => {
    await expect(runAgentMatchAsync({
      scenario: rangerKitesFighterScenario(),
      seed: 1,
      redAgent: 'battlecast.kiting',
      blueAgent: 'lk-47.pass',
    })).rejects.toThrow('lk-47 agents play only with llmActionSpace actual-actions-v1');
    expect(() => runAgentMatch({
      scenario: rangerKitesFighterScenario(),
      seed: 1,
      redAgent: 'battlecast.kiting',
      blueAgent: 'lk-47.pass',
    })).toThrow('lk-47.pass requires runAgentMatchAsync');
  });

  it('plays a hero mirror through the same stepwise menus as an LLM, choosing only from each menu', async () => {
    const match = await runAgentMatchAsync({
      scenario: statusPressureHeroMirrorScenario,
      seed: 1,
      redAgent: 'lk-47.pass',
      blueAgent: 'battlecast.smart',
      maxRounds: 50,
      llmActionSpace: 'actual-actions-v1',
    });
    const turnStarts = match.replay.filter((event) =>
      event.type === 'turn_started' && event.controller?.mode === 'lk-47'
    );
    const resolutions = match.replay.filter((event) =>
      event.type === 'action_resolved' && event.agentId === 'lk-47.pass'
    );

    expect(turnStarts.length).toBeGreaterThan(0);
    expect(turnStarts.every((event) => event.type === 'turn_started' && event.actionSpace === 'actual-actions-v1')).toBe(true);
    expect(resolutions.length).toBeGreaterThan(0);
    expect(offMenuResolutions(match.replay, 'lk-47.pass')).toEqual([]);
    expect(resolutions.every((event) =>
      event.type === 'action_resolved' &&
      (event.acceptedAction.type === 'end_turn' || event.acceptedAction.type === 'reaction')
    )).toBe(true);
    expect(match.winner).toBe('blue');
  });

  it('is asked for an opportunity attack when an enemy leaves its reach', async () => {
    const match = await runAgentMatchAsync({
      scenario: rangerKitesFighterScenario(),
      seed: 1,
      redAgent: 'battlecast.kiting',
      blueAgent: 'lk-47.pass',
      maxRounds: 1,
      llmActionSpace: 'actual-actions-v1',
    });
    const reaction = match.replay.find((event) =>
      event.type === 'action_resolved' &&
      event.acceptedAction.type === 'reaction' &&
      event.acceptedAction.reaction === 'opportunity_attack'
    );

    expect(reaction?.type === 'action_resolved' && reaction.agentId).toBe('lk-47.pass');
    expect(reaction?.type === 'action_resolved' && reaction.logs.some((log) => log.action === 'Opportunity Attack')).toBe(true);
  });

  it('is asked for a damage reaction at the moment it is hit', async () => {
    const match = await runAgentMatchAsync({
      scenario: fighterThreatensRogueScenario(),
      seed: 1,
      redAgent: 'battlecast.aggressive',
      blueAgent: 'lk-47.pass',
      maxRounds: 1,
      llmActionSpace: 'actual-actions-v1',
    });
    const reaction = match.replay.find((event) =>
      event.type === 'action_resolved' &&
      event.acceptedAction.type === 'reaction' &&
      event.acceptedAction.reaction === 'uncanny_dodge'
    );

    expect(reaction?.type === 'action_resolved' && reaction.agentId).toBe('lk-47.pass');
    expect(reaction?.type === 'action_resolved' && reaction.logs.some((log) => log.action === 'Uncanny Dodge')).toBe(true);
  });

  it('stops the match when a policy picks an action that is not on the menu', async () => {
    vi.spyOn(lk47Policies['lk-47.pass'], 'chooseAction').mockResolvedValue({ id: 'attack:nothing:nobody', type: 'end_turn' } as unknown as LegalAction);

    await expect(runAgentMatchAsync({
      scenario: rangerKitesFighterScenario(),
      seed: 1,
      redAgent: 'battlecast.kiting',
      blueAgent: 'lk-47.pass',
      maxRounds: 1,
      llmActionSpace: 'actual-actions-v1',
    })).rejects.toThrow('lk-47.pass chose attack:nothing:nobody, which is not in the legal menu');
  });

  it('replays identically for the same seed', async () => {
    const spec = {
      scenario: statusPressureHeroMirrorScenario,
      seed: 2,
      redAgent: 'battlecast.smart',
      blueAgent: 'lk-47.pass',
      maxRounds: 50,
      llmActionSpace: 'actual-actions-v1',
    } as const;
    const first = await runAgentMatchAsync(spec);
    const second = await runAgentMatchAsync(spec);

    expect(second.finalStateHash).toBe(first.finalStateHash);
    expect(second.replay).toEqual(first.replay);
  });
});

describe('LK-47 bot imitation', () => {
  it('plans a bot turn on a private copy without touching the match dice or the live state', async () => {
    const { state, active } = atTurnStart(statusPressureHeroMirrorScenario, (battle) =>
      battle.creatures.find((creature) => creature.id === battle.initiativeOrder[0])
    );
    const before = structuredClone(state);
    const matchDice = createRng(99);
    await withBattlecastRngAsync(matchDice, async () => {
      const position = matchDice.snapshot();
      const plan = planBotTurn(state, active.id, 'smart', 7);
      expect(plan.length).toBeGreaterThan(0);
      expect(matchDice.snapshot()).toBe(position);
    });
    expect(state).toEqual(before);
    expect(planBotTurn(state, active.id, 'smart', 7)).toEqual(planBotTurn(state, active.id, 'smart', 7));
  });

  it('reads weapon attacks with the target the bot hit', () => {
    const { state, active } = atTurnStart(adjacentFightersScenario(), (battle) =>
      battle.creatures.find((creature) => creature.team === 'red')
    );
    const enemy = state.creatures.find((creature) => creature.team === 'blue');
    const attacks = planBotTurn(state, active.id, 'smart', 3).filter((step) => step.kind === 'attack');

    expect(attacks.length).toBeGreaterThanOrEqual(2);
    expect(attacks.every((step) => step.kind === 'attack' && step.targetId === enemy?.id && step.actionName === 'Longsword')).toBe(true);
  });

  it('reads a heal with the creature it healed', () => {
    const { state, active } = atTurnStart(clericBesideWoundedFighterScenario(), (battle) => {
      const fighter = battle.creatures.find((creature) => creature.team === 'red' && creature.monsterData.heroClass === 'Fighter');
      if (fighter) fighter.currentHp = 4;
      return battle.creatures.find((creature) => creature.monsterData.heroClass === 'Cleric');
    });
    const fighter = state.creatures.find((creature) => creature.team === 'red' && creature.monsterData.heroClass === 'Fighter');
    const plan = planBotTurn(state, active.id, 'smart', 5);

    expect(plan).toEqual([
      { kind: 'heal', actionName: 'Healing Word', targetIds: [fighter?.id] },
      { kind: 'heal', actionName: 'Channel Divinity: Preserve Life', targetIds: [fighter?.id] },
    ]);
  });

  it.each(['smart', 'aggressive', 'kiting', 'defensive'] as const)('plays a full match as %s using only menu actions, deterministically', async (tactic) => {
    const spec = {
      scenario: statusPressureHeroMirrorScenario,
      seed: 3,
      redAgent: `lk-47.imitate-${tactic}`,
      blueAgent: `battlecast.${tactic}`,
      maxRounds: 50,
      llmActionSpace: 'actual-actions-v1',
    } as const;
    const first = await runAgentMatchAsync(spec);
    const second = await runAgentMatchAsync(spec);

    expect(first.state.isComplete).toBe(true);
    expect(offMenuResolutions(first.replay, `lk-47.imitate-${tactic}`)).toEqual([]);
    expect(second.finalStateHash).toBe(first.finalStateHash);
  }, 30_000);
});

describe('LK-47 planner', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each(['lk-47', 'lk-47.independent'] as const)('%s plans on private copies without touching the match dice or the live state', async (agentId) => {
    const { state, active } = atTurnStart(adjacentFightersScenario(), (battle) =>
      battle.creatures.find((creature) => creature.team === 'red')
    );
    const turn = { attackRollsRemaining: estimateAttackRollBudget(active), attackActionStarted: false, flurryStrikesRemaining: 0, disengaged: false, ended: false };
    const catalogue = generateLegalActions(state, active, { includeActualActions: true, actualTurnContext: turn });
    const before = structuredClone(state);
    const matchDice = createRng(99);
    await withBattlecastRngAsync(matchDice, async () => {
      const position = matchDice.snapshot();
      const choice = await lk47Policies[agentId].chooseAction({ state, activeCreature: active, catalogue, turn, simulator: lk47Simulator });
      expect(catalogue.actions.map((action) => action.id)).toContain(choice.id);
      expect(matchDice.snapshot()).toBe(position);
    });
    expect(state).toEqual(before);
  });

  it('runs no bot turn in its simulations when independent, and moves distant creatures itself', async () => {
    const { state, active } = atTurnStart(clericBesideWoundedFighterScenario(), (battle) =>
      battle.creatures.find((creature) => creature.monsterData.heroClass === 'Cleric')
    );
    const enemy = state.creatures.find((creature) => creature.team === 'blue');
    const turn = { attackRollsRemaining: estimateAttackRollBudget(active), attackActionStarted: false, flurryStrikesRemaining: 0, disengaged: false, ended: false };
    const context = { state, activeCreature: active, catalogue: generateLegalActions(state, active, { includeActualActions: true, actualTurnContext: turn }), turn, simulator: lk47Simulator };
    const scripted = vi.spyOn(lk47Simulator, 'playScriptedTurn');
    const applied = vi.spyOn(lk47Simulator, 'applyAction');

    await lk47Policies['lk-47.independent'].chooseAction(context);
    expect(scripted).not.toHaveBeenCalled();
    expect(applied.mock.calls.some(([, mover, action]) => mover.id === enemy?.id && action.type === 'move_toward')).toBe(true);

    await lk47Policies['lk-47'].chooseAction(context);
    expect(scripted).toHaveBeenCalled();
  });

  it.each(['lk-47', 'lk-47.independent'] as const)('%s plays a match using only menu actions, deterministically', async (agentId) => {
    const spec = {
      scenario: adjacentFightersScenario(),
      seed: 4,
      redAgent: agentId,
      blueAgent: 'battlecast.smart',
      maxRounds: 2,
      llmActionSpace: 'actual-actions-v1',
    } as const;
    const first = await runAgentMatchAsync(spec);
    const second = await runAgentMatchAsync(spec);

    expect(offMenuResolutions(first.replay, agentId)).toEqual([]);
    expect(first.replay.some((event) => event.type === 'action_resolved' && event.agentId === agentId)).toBe(true);
    expect(second.finalStateHash).toBe(first.finalStateHash);
  }, 60_000);
});

// Builds a battle the way the match runner does and processes one creature's turn start, as the runner does
// before an actual-action agent's first decision. `prepare` may adjust the battle and returns that creature.
function atTurnStart(
  scenario: D20benchScenario,
  prepare: (state: BattleState) => Creature | undefined,
): { state: BattleState; active: Creature } {
  return withBattlecastRng(createRng(1), () => {
    const state = initBattle(createBattlecastCreatures(scenario.combatants, true), scenario.gridSize);
    const map = maps.find((candidate) => candidate.id === scenario.mapId);
    state.terrainBlocked = buildMovementBlockedSet(map?.terrain);
    state.terrainSightBlocked = buildSightBlockedSet(map?.terrain);
    const active = prepare(state);
    if (!active) throw new Error('test fixture found no active creature');
    beginBattlecastControlledTurn(state, active, { autoClassFeatures: false });
    return { state, active };
  });
}

// Each accepted action must appear in the menu it answers. Reaction menus open inside a step (an opportunity
// attack during a move can prompt Cutting Words), so menus are matched like brackets, per creature.
function offMenuResolutions(replay: ReplayEvent[], agentId: string): string[] {
  const openMenus = new Map<string, Array<Set<string>>>();
  const offMenu: string[] = [];
  for (const event of replay) {
    if (event.type === 'turn_started') {
      const stack = openMenus.get(event.activeCreatureId) ?? [];
      stack.push(new Set(event.legalActions.map((action) => action.id)));
      openMenus.set(event.activeCreatureId, stack);
    } else if (event.type === 'action_resolved') {
      const menu = openMenus.get(event.activeCreatureId)?.pop();
      if (event.agentId === agentId && !menu?.has(event.acceptedAction.id)) {
        offMenu.push(`${event.activeCreatureId}:${event.acceptedAction.id}`);
      }
    }
  }
  const unclosed = [...openMenus.values()].reduce((total, stack) => total + stack.length, 0);
  if (unclosed > 0) offMenu.push(`${unclosed} menus never answered`);
  return offMenu;
}

function rangerKitesFighterScenario(): D20benchScenario {
  return {
    id: 'test.lk-47.ranger-kites-fighter.v1',
    name: 'Ranger Kites Fighter',
    description: 'A kiting ranger backs away from an LK-47 fighter, triggering an opportunity attack prompt.',
    battleType: 'duel-smoke',
    visibility: 'hidden',
    rulesetId: 'test-rules',
    dataPackId: 'test-data',
    scenarioVersion: '1.0.0',
    gridSize: 8,
    tacticalTags: ['test'],
    designNotes: ['test fixture'],
    combatants: [
      { monster: buildHero('Ranger', 5), team: 'red', position: { x: 2, y: 2 } },
      { monster: buildHero('Fighter', 5), team: 'blue', position: { x: 2, y: 3 } },
    ],
  };
}

function fighterThreatensRogueScenario(): D20benchScenario {
  return {
    id: 'test.lk-47.fighter-threatens-rogue.v1',
    name: 'Fighter Threatens Rogue',
    description: 'A scripted fighter attacks an LK-47 rogue that can spend Uncanny Dodge.',
    battleType: 'reaction-smoke',
    visibility: 'hidden',
    rulesetId: 'test-rules',
    dataPackId: 'test-data',
    scenarioVersion: '1.0.0',
    gridSize: 8,
    tacticalTags: ['test'],
    designNotes: ['test fixture'],
    combatants: [
      { monster: buildHero('Fighter', 20), team: 'red', position: { x: 2, y: 2 } },
      { monster: buildHero('Rogue', 5), team: 'blue', position: { x: 2, y: 3 } },
    ],
  };
}

function adjacentFightersScenario(): D20benchScenario {
  return {
    id: 'test.lk-47.adjacent-fighters.v1',
    name: 'Adjacent Fighters',
    description: 'Two adjacent level-5 fighters; the red one acts first in the plan test.',
    battleType: 'duel-smoke',
    visibility: 'hidden',
    rulesetId: 'test-rules',
    dataPackId: 'test-data',
    scenarioVersion: '1.0.0',
    gridSize: 8,
    tacticalTags: ['test'],
    designNotes: ['test fixture'],
    combatants: [
      { monster: buildHero('Fighter', 5), team: 'red', position: { x: 2, y: 2 } },
      { monster: buildHero('Fighter', 5), team: 'blue', position: { x: 2, y: 3 } },
    ],
  };
}

function clericBesideWoundedFighterScenario(): D20benchScenario {
  return {
    id: 'test.lk-47.cleric-beside-wounded-fighter.v1',
    name: 'Cleric Beside Wounded Fighter',
    description: 'A red cleric stands next to a red fighter the test leaves at 4 HP; the enemy is far away.',
    battleType: 'duel-smoke',
    visibility: 'hidden',
    rulesetId: 'test-rules',
    dataPackId: 'test-data',
    scenarioVersion: '1.0.0',
    gridSize: 10,
    tacticalTags: ['test'],
    designNotes: ['test fixture'],
    combatants: [
      { monster: buildHero('Cleric', 5), team: 'red', position: { x: 2, y: 2 } },
      { monster: buildHero('Fighter', 5), team: 'red', position: { x: 2, y: 3 } },
      { monster: buildHero('Fighter', 5), team: 'blue', position: { x: 8, y: 8 } },
    ],
  };
}
