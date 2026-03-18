# sioux_falls_network.py

from flow.networks.base import Network
from flow.core.params import InitialConfig, TrafficLightParams

class SiouxFallsNetwork(Network):
    """Network class that wraps the Sioux Falls SUMO template.

    All geometry, routes, and vehicle types are read from the SUMO
    XML files via NetParams(template=...).
    """

    def __init__(self, name, vehicles, net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        # You don’t need to do anything special here;
        # just pass things up to the base class.
        super().__init__(
            name=name,
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=traffic_lights
        )

    # NOTE:
    # - We do NOT override specify_nodes, specify_edges, specify_routes, ...
    #   because the base Network class will read them from the SUMO templates
    #   when net_params.template is provided.
    # - You can override specify_edge_starts later if you want a custom
    #   1D “unrolled” coordinate system, but it’s optional.
