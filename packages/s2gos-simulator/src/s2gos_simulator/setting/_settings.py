from s2gos_utils.setting import settings as util_settings

# Validate Simulator config
# util_settings.validators.register(
#     Validator("simulator.sim_test", must_exist=True), # Add Validators here
# )
# util_settings.validators.validate(only="simulator")


# Forward s2gos_utils settings
settings = util_settings
