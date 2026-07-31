class BaseArticulationBuilder:
    pass


class ArticulationBuilder(BaseArticulationBuilder):
    """A multi-backend articulation builder."""

    def __init__(
        self,
        physics_builder: BaseArticulationBuilder,
        articulation_builder: BaseArticulationBuilder,
    ) -> None:
        self.builder = physics_builder
        self.articulation_builder = articulation_builder
