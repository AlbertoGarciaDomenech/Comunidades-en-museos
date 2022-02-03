from gephistreamer import streamer, graph

class GephiVisualization:

    def __init__(self, hostname='localhost', port=8080, workspace='workspace0'):
        self.stream = streamer.Streamer(streamer.GephiWS(hostname=hostname, port=port, workspace=workspace))

    def load_community(self, users, distances, users_properties=None):
        """Method to visualizate communities by users in Gephi.

        Args:
            users (array): Array that contains information of users. First value of user should be ID,
            rest information is incorporated to Node.
            distances (array): Matrix that contains the similarity/distance between users.
        """

        # Cargamos los usuarios como nodos en Gephi
        self.nodes = {}

        for u in users:
            node = graph.Node(u[0], size=100)

            if users_properties != None:
                for i in range(1, len(u)):
                    node.property[users_properties[i-1]] = u[i]

            self.nodes[u[0]] = node
            self.stream.add_node(node)

        # Cargamos las distancias entre usuarios como ejes
        self.edges = []
        for i in  range(len(distances)):
            for j in range(len(distances[i])):

                if i != j:
                    source = self.nodes[users[i][0]]
                    target = self.nodes[users[j][0]]
                    edge = graph.Edge(source, target, directed=False, weight=distances[i][j])
                    self.edges.append(edge)
                    self.stream.add_edge(edge)

    def clean_graph(self):

        # Eliminamos todos los ejes
        for e in self.edges:
            self.stream.delete_edge(e)

        # Eliminamos todos los nodos
        for n in self.nodes:
            self.stream.delete_node(self.nodes[n])

        self.edges.clear()
        self.nodes.clear()