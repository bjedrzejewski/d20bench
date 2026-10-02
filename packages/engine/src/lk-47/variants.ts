import type { ActualTurnContext } from '../agent-match.js';
import type { BattleState, TacticType } from '../battlecast/engine/combat.js';
import type { Creature } from '../battlecast/types/monster.js';
import type { Lk47AgentId } from '../agents.js';
import type { LegalAction, LegalActionCatalogue } from '../legal-actions.js';
import type { SeededRng } from '../random.js';
import { createImitatorPolicy } from './imitator.js';
import { createPlannerPolicy } from './planner.js';

// Applies the match's own rules to private copies of a battle; supplied by the match runner.
export interface Lk47Simulator {
  // The legal menu without movement squares, which are by far its most expensive part.
  actionsInPlace(state: BattleState, active: Creature, turn: ActualTurnContext): LegalActionCatalogue;
  applyAction(state: BattleState, active: Creature, action: LegalAction, turn: ActualTurnContext, rng: SeededRng): Promise<boolean>;
  finishTurn(state: BattleState, active: Creature): void;
  playScriptedTurn(state: BattleState, active: Creature, tactic: TacticType): void;
  // Starts a turn as the harness does for a menu-driven agent; undefined if the creature cannot act.
  beginTurn(state: BattleState, active: Creature): ActualTurnContext | undefined;
}

export interface Lk47DecisionContext {
  state: BattleState;
  activeCreature: Creature;
  catalogue: LegalActionCatalogue;
  turn?: ActualTurnContext;
  simulator: Lk47Simulator;
}

export interface Lk47Policy {
  chooseAction(context: Lk47DecisionContext): Promise<LegalAction>;
}

const schedule = [{ samples: 1, keep: 6 }, { samples: 4, keep: 2 }, { samples: 12, keep: 1 }];

export const lk47Policies: Record<Lk47AgentId, Lk47Policy> = {
  // The final agent from stage 5 on: four times every sample count of the stage 2 schedule.
  'lk-47': createPlannerPolicy({
    model: 'smart',
    schedule: schedule.map(({ samples, keep }) => ({ samples: samples * 4, keep })),
  }),
  // Stage 7 candidate for a second season entry: four times the samples of `lk-47`.
  'lk-47.16x': createPlannerPolicy({
    model: 'smart',
    schedule: schedule.map(({ samples, keep }) => ({ samples: samples * 16, keep })),
  }),
  // Stage 3 budget levels below the final one; stages 2 and 3 called the 1x schedule `lk-47`.
  'lk-47.budget-min': createPlannerPolicy({ model: 'smart', schedule: [{ samples: 1, keep: 1 }] }),
  'lk-47.budget-1x': createPlannerPolicy({ model: 'smart', schedule }),
  // Stage 6b: no Battlecast bot code in the simulations; every simulated creature plays LK-47's own greedy policy.
  'lk-47.independent': createPlannerPolicy({ model: 'greedy', schedule }),
  // Integrity probe: never acts on its own turn, takes the first reaction offered.
  // A side played this way must lose almost every match; if not, the harness is wrong.
  'lk-47.pass': {
    async chooseAction({ catalogue }) {
      const reaction = catalogue.actions.find((action) => action.type === 'reaction' && action.reaction !== 'decline');
      return reaction ?? catalogue.actions.find((action) => action.type === 'end_turn') ?? catalogue.actions[0];
    },
  },
  // Replays each scripted bot's own turn through the legal menu; stage 1 measures what the menu loses.
  'lk-47.imitate-smart': createImitatorPolicy('smart'),
  'lk-47.imitate-aggressive': createImitatorPolicy('aggressive'),
  'lk-47.imitate-kiting': createImitatorPolicy('kiting'),
  'lk-47.imitate-defensive': createImitatorPolicy('defensive'),
};
