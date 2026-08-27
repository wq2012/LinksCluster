"""Graph data structure for maintaining subcluster connectivity."""

import collections
import typing


class SubclusterGraph:
  """Undirected graph representing connections between subclusters."""

  def __init__(self):
    """Initialize an empty graph."""
    self.adj: typing.Dict[int, typing.Set[int]] = collections.defaultdict(set)

  def add_node(self, node_id: int):
    """Add a node to the graph if not already present."""
    if node_id not in self.adj:
      self.adj[node_id] = set()

  def remove_node(self, node_id: int):
    """Remove a node and all its incident edges."""
    if node_id in self.adj:
      for neighbor in list(self.adj[node_id]):
        self.adj[neighbor].discard(node_id)
      del self.adj[node_id]

  def add_edge(self, u: int, v: int):
    """Add an undirected edge between u and v."""
    self.adj[u].add(v)
    self.adj[v].add(u)

  def remove_edge(self, u: int, v: int):
    """Remove the undirected edge between u and v if present."""
    if u in self.adj:
      self.adj[u].discard(v)
    if v in self.adj:
      self.adj[v].discard(u)

  def has_edge(self, u: int, v: int) -> bool:
    """Return True if edge (u, v) exists."""
    return u in self.adj and v in self.adj[u]

  def neighbors(self, u: int) -> typing.Set[int]:
    """Return the set of neighbors of node u."""
    return self.adj.get(u, set())

  def degree(self, u: int) -> int:
    """Return the degree of node u."""
    return len(self.adj.get(u, set()))

  def merge_nodes(self, keep_id: int, remove_id: int):
    """Merge remove_id into keep_id, preserving all neighbor connections."""
    if remove_id not in self.adj:
      return
    remove_neighbors = set(self.adj[remove_id])
    self.remove_node(remove_id)
    for neighbor in remove_neighbors:
      if neighbor != keep_id:
        self.add_edge(keep_id, neighbor)

  def check_severed(
      self,
      start_id: int,
      target_id: int,
      allowed_nodes: typing.Optional[typing.Set[int]] = None
  ) -> typing.Tuple[bool, typing.Set[int]]:
    """Check if removing an edge disconnected start_id from target_id.

    Traverses the component of start_id using BFS. If target_id is reached,
    they remain connected. If target_id is unreachable, returns True and the
    set of nodes reachable from start_id.

    Args:
      start_id: the node to begin BFS from
      target_id: the node we want to test reachability to
      allowed_nodes: optional set of nodes restricting traversal

    Returns:
      severed: bool, True if start_id cannot reach target_id
      visited: set of nodes reachable from start_id (empty if not severed)
    """
    visited: typing.Set[int] = set()
    queue = collections.deque([start_id])
    visited.add(start_id)

    while queue:
      curr = queue.popleft()
      if curr == target_id:
        return False, set()
      for nbr in self.adj.get(curr, ()):
        if nbr not in visited:
          if allowed_nodes is None or nbr in allowed_nodes:
            visited.add(nbr)
            queue.append(nbr)

    return True, visited

  def get_component(
      self,
      start_id: int,
      allowed_nodes: typing.Optional[typing.Set[int]] = None
  ) -> typing.Set[int]:
    """Return all nodes in the connected component of start_id."""
    visited: typing.Set[int] = set()
    queue = collections.deque([start_id])
    visited.add(start_id)

    while queue:
      curr = queue.popleft()
      for nbr in self.adj.get(curr, ()):
        if nbr not in visited:
          if allowed_nodes is None or nbr in allowed_nodes:
            visited.add(nbr)
            queue.append(nbr)

    return visited
