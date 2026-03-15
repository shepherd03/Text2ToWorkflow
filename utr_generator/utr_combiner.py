import re

from .schema import Action, ControlIntent, Resource, UTR, Variable


class UTRCombiner:
    def combine(
        self,
        actions: list[Action],
        resources: list[Resource],
        control_intents: list[ControlIntent],
        variables: list[Variable],
    ) -> UTR:
        ordered_actions = sorted(actions, key=lambda x: x.order)
        for index, action in enumerate(ordered_actions, start=1):
            action.action_id = f"act_{index}"
            action.order = index
            if not action.description:
                action.description = action.action_name.replace("_", " ")
        for index, resource in enumerate(resources, start=1):
            resource.resource_id = f"res_{index}"
            if not resource.description:
                resource.description = resource.name
        action_name_map = {action.action_name: action.action_id for action in ordered_actions}
        valid_action_ids = {action.action_id for action in ordered_actions}
        for index, intent in enumerate(control_intents, start=1):
            intent.intent_id = f"intent_{index}"
            mapped_targets: list[str] = []
            for target in intent.target_actions:
                candidate = action_name_map.get(target, target)
                if candidate in valid_action_ids:
                    mapped_targets.append(candidate)
                elif re.fullmatch(r"act_\d+", candidate) and candidate in valid_action_ids:
                    mapped_targets.append(candidate)
            intent.target_actions = mapped_targets
        for index, variable in enumerate(variables, start=1):
            variable.var_id = f"var_{index}"
            if not variable.source:
                variable.source = "user_input"
        return UTR(
            actions=ordered_actions,
            resources=resources,
            control_intents=control_intents,
            variables=variables,
        )
