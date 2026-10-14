"""The tutorial's MCM limonene mechanism with its rate constants at one temperature, pressure and RH."""
from __future__ import annotations


class Chemistry:
    """Parsed mechanism (``.mechanism``), air composition (``.air``) and rate constants.

    ``k[label]`` is the thermal rate constant of a reaction (1/s for one reactant,
    cm3 molec-1 s-1 for two); ``ro2_dependent[label]`` is True if it is also multiplied by
    the RO2 sum. Photolysis reactions get their J during a run.
    """

    def __init__(self, temp_k: float, pres_pa: float, rh: float):
        from tomas_jax.box.coupling import tutorial_mechanism
        from tomas_jax.gaschem.environment import air_environment
        from tomas_jax.gaschem.mechanism import RateModel

        self.mechanism = tutorial_mechanism()      # the .eqn file plus the N2O5-uptake edit (A1)
        self.air = air_environment(temp_k, pres_pa, rh)   # M, O2, N2, H2O [molec/cm3]
        k_thermal, ro2_dependent, _ = RateModel(self.mechanism).at_env(self.air)
        labels = [r.label for r in self.mechanism.reactions]
        self.k = dict(zip(labels, (float(v) for v in k_thermal)))
        self.ro2_dependent = dict(zip(labels, (bool(v) for v in ro2_dependent)))

    def reactions_of(self, species: str, role: str = "reactant") -> list:
        """Reactions with ``species`` as a reactant (role="reactant") or product (role="product")."""
        if role not in ("reactant", "product"):
            raise ValueError('role must be "reactant" or "product"')
        if species not in self.mechanism.species:
            raise ValueError(f"{species!r} is not a species of the mechanism")
        return [r for r in self.mechanism.reactions
                if species in (r.reactants if role == "reactant" else [p for p, _ in r.products])]

    def show_reactions(self, species: str, role: str = "reactant") -> None:
        """Print every reaction with ``species`` as a reactant or product, with its rate constant."""
        for r in self.reactions_of(species, role):
            lhs = " + ".join(r.reactants) + (" + hν" if r.photolysis else "")
            rhs = " + ".join(p if c == 1 else f"{c:g} {p}" for p, c in r.products) or "(not tracked)"
            if r.photolysis:
                rate = "J (photolysis)"
            elif self.ro2_dependent[r.label]:
                rate = f"k = {self.k[r.label]:.3g} cm³ molec⁻¹ s⁻¹ × [RO2]"
            else:
                unit = "s⁻¹" if len(r.reactants) == 1 else "cm³ molec⁻¹ s⁻¹"
                rate = f"k = {self.k[r.label]:.3g} {unit}"
            print(f"<{r.label}> {lhs} → {rhs}    {rate}")

    def formula(self, species: str) -> str:
        """Molecular formula such as "C10H16" from the mechanism's atom counts."""
        atoms = self.mechanism.atoms[species]
        return "".join(f"{a}{atoms[a] if atoms[a] > 1 else ''}"
                       for a in ("C", "H", "N", "O", "S") if atoms.get(a))


def mechanism(scenario="level2_limonene_soa") -> Chemistry:
    """The mechanism with rate constants at the temperature, pressure and RH of ``scenario``
    (anything run() accepts: preset name, Scenario, RunResult or dict)."""
    from .model import scenario_from
    env = scenario_from(scenario).environment
    return Chemistry(env.temp_k, env.pres_pa, env.rh)
