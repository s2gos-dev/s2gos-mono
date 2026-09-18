from procodile import ProcessRegistry

#: Free-tier processes only. Kept apart from ``s2gos_apps.registry`` so that serving
#: the free tier does not import the generator/simulator process modules.
registry = ProcessRegistry()
