import type { BattleState } from '../battlecast/engine/combat.js';

// Shares nothing mutable with the live battle. Logs and events start empty: the rules read past events only
// within the current turn, and LK-47 copies a battle only between its own decisions.
export function copyBattle(state: BattleState): BattleState {
  const { terrainBlocked, terrainSightBlocked, damageReactionHooks: _hooks, logs: _logs, events: _events, ...rest } = state;
  const copy = structuredClone(rest) as BattleState;
  copy.logs = [];
  copy.events = [];
  copy.terrainBlocked = terrainBlocked;
  copy.terrainSightBlocked = terrainSightBlocked;
  return copy;
}
