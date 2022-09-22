import random
import faker.generator


class Random(random.Random):
    state_set: bool = False


randgen = Random()


def get_random_state():
    """Retrieve the state of factory.fuzzy's random generator."""
    state = randgen.getstate()
    # Returned state must represent both Faker and factory_boy.
    faker.generator.random.setstate(state)
    return state


def set_random_state(state):
    """Force-set the state of factory.fuzzy's random generator."""
    randgen.state_set = True
    randgen.setstate(state)

    faker.generator.random.setstate(state)


def reseed_random(seed):
    """Reseed factory.fuzzy's random generator."""
    r = Random(seed)
    random_internal_state = r.getstate()
    set_random_state(random_internal_state)
