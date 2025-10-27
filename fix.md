Vector index error after compute_rule_fure_levels is called on the reworked fuzzysystem

Can be a cpp indexing error but dont know why

rules_stat_activations calls _compute_rule_fire_levels which fails at 
            values = fuzzy_system.compute_rules_fire_levels(sample)

We can try to see if this methods works before any renaming aka everytime on a fuzzysystem
