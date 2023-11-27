from hypothesis import settings, Phase
values = settings.get_profile("default").__dict__ | {"print_blob": True}
settings.register_profile("dev", settings(**values))
settings.register_profile("failfast", phases=[Phase.generate], print_blob=True)

# TODO: switch to something like this:
# settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
settings.load_profile("dev")
