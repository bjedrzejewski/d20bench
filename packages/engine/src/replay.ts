import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import type { BattleState, BattleLog } from './battlecast/engine/combat.js';
import type { TacticType } from './battlecast/engine/combat.js';
import type { AnimationEvent } from './battlecast/types/animation.js';
import type { AgentId } from './agents.js';
import type { LegalActionCatalogue, LegalAction } from './legal-actions.js';
import type { OpenRouterDecisionTrace } from './openrouter-agent.js';
import type { D20benchScenario } from './scenario.js';

export type ReplayEventController =
  | {
      mode: 'legal-action';
      agentId: AgentId;
    }
  | {
      mode: 'battlecast-tactic';
      agentId: AgentId;
      tactic: TacticType;
    }
  | {
      mode: 'openrouter-llm';
      agentId: AgentId;
      model: string;
    }
  | {
      mode: 'lk-47';
      agentId: AgentId;
    };

export type ReplayEvent =
  | {
      type: 'match_started';
      matchId: string;
      scenario: Pick<D20benchScenario, 'id' | 'rulesetId' | 'dataPackId' | 'scenarioVersion'>;
      seed: string | number;
      redAgent: AgentId;
      blueAgent: AgentId;
      stateHash: string;
    }
  | {
      type: 'round_started';
      matchId: string;
      round: number;
      stateHash: string;
    }
  | {
      type: 'turn_started';
      matchId: string;
      round: number;
      turnIndex: number;
      turnStep?: number;
      activeCreatureId: string;
      activeCreatureName: string;
      controller?: ReplayEventController;
      actionSpace?: LegalActionCatalogue['actionSpace'];
      legalActions: LegalActionCatalogue['actions'];
      actionEconomy?: LegalActionCatalogue['actionEconomy'];
      stateHash: string;
    }
  | {
      type: 'action_resolved';
      matchId: string;
      round: number;
      turnIndex: number;
      turnStep?: number;
      activeCreatureId: string;
      agentId: AgentId;
      requestedActionId: string;
      acceptedAction: LegalAction;
      llmTrace?: OpenRouterDecisionTrace;
      logs: BattleLog[];
      events: AnimationEvent[];
      stateHash: string;
    }
  | {
      type: 'round_ended';
      matchId: string;
      round: number;
      stateHash: string;
    }
  | {
      type: 'match_finished';
      matchId: string;
      winner: BattleState['winner'];
      finalStateHash: string;
      rounds: number;
    };

export async function writeReplayJsonl(path: string, events: ReplayEvent[]): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${events.map((event) => JSON.stringify(event)).join('\n')}\n`, 'utf8');
}

export async function readReplayJsonl(path: string): Promise<ReplayEvent[]> {
  const text = await readFile(path, 'utf8');
  return text
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as ReplayEvent);
}

export function verifyReplayStructure(events: ReplayEvent[]): { ok: boolean; errors: string[] } {
  const errors: string[] = [];
  if (events.length === 0) {
    return { ok: false, errors: ['replay is empty'] };
  }
  if (events[0]?.type !== 'match_started') {
    errors.push('first replay event must be match_started');
  }
  const last = events[events.length - 1];
  if (last?.type !== 'match_finished') {
    errors.push('last replay event must be match_finished');
  }
  const matchIds = new Set(events.map((event) => event.matchId));
  if (matchIds.size !== 1) {
    errors.push(`expected one match id, found ${matchIds.size}`);
  }
  for (const event of events) {
    const hash = 'stateHash' in event ? event.stateHash : event.type === 'match_finished' ? event.finalStateHash : undefined;
    if (hash && !/^[a-f0-9]{64}$/.test(hash)) {
      errors.push(`invalid state hash on ${event.type}: ${hash}`);
    }
  }
  return { ok: errors.length === 0, errors };
}
