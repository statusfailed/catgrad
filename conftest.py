from hypothesis import settings, Phase
settings.register_profile("failfast", phases=[Phase.generate])
