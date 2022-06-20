import logging

from rxn_chemutils.exceptions import InvalidSmiles
from rxn_chemutils.reaction_equation import (
    merge_reactants_and_agents, sort_compounds, canonicalize_compounds, remove_duplicate_compounds
)
from rxn_chemutils.reaction_smiles import parse_any_reaction_smiles

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())


def standardize_for_fp_model(rxn: str) -> str:
    """
    Standardize a reaction SMILES for input to rxnfp model.
    """

    try:
        return standardize_rxn_smiles(
            rxn_smiles=rxn,
            fragment_bond='.',
            ordered_precursors=True,
            canonicalize=True,
            remove_duplicate_molecules=True
        )
    except InvalidSmiles as e:
        logger.warning(
            f'Error during canonicalization of "{e.smiles}"; doing reordering and fragment bond replacement only.'
        )
        return standardize_rxn_smiles(
            rxn_smiles=rxn,
            fragment_bond='.',
            ordered_precursors=True,
            canonicalize=False,
            remove_duplicate_molecules=True
        )


def standardize_rxn_smiles(
    rxn_smiles: str, fragment_bond: str, ordered_precursors: bool, canonicalize: bool,
    remove_duplicate_molecules: bool
) -> str:
    """
    Standardize a reaction SMILES with multiple options.

    Args:
        rxn_smiles: reaction SMILES to standardize.
        fragment_bond: fragment bond to use in the produced string.
        ordered_precursors: whether to order the compounds alphabetically.
        canonicalize: whether to canonicalize all the SMILES strings.
        remove_duplicate_molecules: whether to remove duplicate molecules.

    Returns:
        Standardized SMILES string.
    """

    reaction_equation = parse_any_reaction_smiles(rxn_smiles)
    reaction_equation = merge_reactants_and_agents(reaction_equation)

    if canonicalize:
        reaction_equation = canonicalize_compounds(reaction_equation)

    if ordered_precursors:
        reaction_equation = sort_compounds(reaction_equation)

    if remove_duplicate_molecules:
        reaction_equation = remove_duplicate_compounds(reaction_equation)

    return reaction_equation.to_string(fragment_bond=fragment_bond)
