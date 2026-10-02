import { getMeleeActions } from '../battlecast/engine/ai-targeting.js';
import type { Creature } from '../battlecast/types/monster.js';
import type { LegalAction, LegalActionCatalogue } from '../legal-actions.js';

// Scripted bots always react (first melee weapon for opportunity attacks) and smite every hit (free smite, else lowest slot).
export function chooseBotReaction(reactor: Creature, catalogue: LegalActionCatalogue): LegalAction | undefined {
  if (!catalogue.actions.some((action) => action.type === 'reaction')) return undefined;
  const uses = catalogue.actions.filter((action) => action.type === 'reaction' && action.reaction !== 'decline');
  const firstMelee = getMeleeActions(reactor)[0]?.name;
  return uses.find((action) => action.type === 'reaction' && action.actionName === firstMelee) ?? uses[0] ?? catalogue.actions[0];
}

export function chooseBotSmite(catalogue: LegalActionCatalogue): LegalAction | undefined {
  const smites = catalogue.actions.filter((action): action is Extract<LegalAction, { type: 'smite' }> => action.type === 'smite');
  if (smites.length === 0) return undefined;
  const free = smites.find((action) => action.resourceKey === 'free-divine-smite');
  const slots = smites
    .filter((action) => action.smite === 'divine_smite' && action.resourceKey !== 'free-divine-smite')
    .sort((left, right) => (left.slotLevel ?? 0) - (right.slotLevel ?? 0));
  return free ?? slots[0] ?? smites.find((action) => action.smite === 'decline') ?? smites[0];
}
