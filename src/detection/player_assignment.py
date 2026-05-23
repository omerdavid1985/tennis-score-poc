from __future__ import annotations

from dataclasses import dataclass

from detection.player_detector import PlayerDetection


@dataclass
class AssignedPlayers:
    """
    Near/far player assignment for one frame.
    """

    near_player: PlayerDetection | None
    far_player: PlayerDetection | None


def assign_near_far_players(
    players: list[PlayerDetection],
) -> AssignedPlayers:
    """
    Assign near-side and far-side players using image position.

    Assumption:
    The camera is behind/near one baseline.

    Therefore:
    - near player usually appears lower in the image
    - far player usually appears higher in the image
    """

    if not players:
        return AssignedPlayers(
            near_player=None,
            far_player=None,
        )

    # Sort by vertical center of bounding box.
    # Smaller y = higher in image = far side.
    # Larger y = lower in image = near side.
    sorted_players = sorted(
        players,
        key=lambda player: _box_center_y(player),
    )

    far_player = sorted_players[0]
    near_player = sorted_players[-1]

    # If only one player is visible, do not pretend we know both.
    if len(sorted_players) == 1:
        return AssignedPlayers(
            near_player=near_player,
            far_player=None,
        )

    return AssignedPlayers(
        near_player=near_player,
        far_player=far_player,
    )


def _box_center_y(
    player: PlayerDetection,
) -> float:
    """
    Vertical center of a player bounding box.
    """

    return 0.5 * (player.y1 + player.y2)