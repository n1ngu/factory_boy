import random
import typing as t

import faker.generator


RandomState = tuple[t.Any, ...]


class Random(random.Random):
    state_set: bool = False


randgen = Random()


def get_random_state() -> RandomState:
    """Retrieve the state of factory.fuzzy's random generator."""
    state = randgen.getstate()
    # Returned state must represent both Faker and factory_boy.
    faker.generator.random.setstate(state)
    return state


def set_random_state(state: RandomState) -> None:
    """Force-set the state of factory.fuzzy's random generator."""
    randgen.state_set = True
    randgen.setstate(state)

    faker.generator.random.setstate(state)


def reseed_random(seed: t.Any) -> None:
    """Reseed factory.fuzzy's random generator."""
    r = Random(seed)
    random_internal_state = r.getstate()
    set_random_state(random_internal_state)
