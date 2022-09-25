# Copyright: See the LICENSE file.

import typing as t

# Strategies
BUILD_STRATEGY = 'build'
CREATE_STRATEGY = 'create'
STUB_STRATEGY = 'stub'

Strategy = t.Literal['build', 'create', 'stub']

#: String for splitting an attribute name into a
#: (subfactory_name, subfactory_field) tuple.
SPLITTER = '__'


# Target build phase, for declarations
class BuilderPhase:
    #: During attribute resolution/computation
    ATTRIBUTE_RESOLUTION = 'attributes'

    #: Once the target object has been built
    POST_INSTANTIATION = 'post_instance'


def get_builder_phase(obj: t.Any) -> t.Literal["attributes", "post_instance", None]:
    return getattr(obj, 'FACTORY_BUILDER_PHASE', None)
