import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

/**
 * Your implementation of various graph traversal algorithms.
 */
public class GraphAlgorithms {

    /**
     * Performs a breadth first search (bfs) on the input graph, starting at
     * the parameterized starting vertex.
     *
     * When exploring a vertex, explore in the order of neighbors returned by
     * the adjacency list. Failure to do so may cause you to lose points.
     *
     * You may import/use java.util.Set, java.util.List, java.util.Queue, and
     * any classes that implement the aforementioned interfaces, as long as they
     * are efficient.
     *
     * The only instance of java.util.Map that you should use is the adjacency
     * list from graph. DO NOT create new instances of Map for BFS
     * (storing the adjacency list in a variable is fine).
     *
     * DO NOT modify the structure of the graph. The graph should be unmodified
     * after this method terminates.
     *
     * You may assume that the passed in start vertex and graph will not be null.
     * You may assume that the start vertex exists in the graph.
     *
     * @param <T>   The generic typing of the data.
     * @param start The vertex to begin the bfs on.
     * @param graph The graph to search through.
     * @return List of vertices in visited order.
     */
    public static <T> List<Vertex<T>> bfs(Vertex<T> start, Graph<T> graph) {
        // WRITE YOUR CODE HERE (DO NOT MODIFY METHOD HEADER)!
        Map<Vertex<T>, List<VertexDistance<T>>> adjList = graph.getAdjList();
        if (!(adjList.containsKey(start))) {
            throw new java.lang.IllegalArgumentException("Start provided does not exist in the graph.");
        }
        List<Vertex<T>> visited = new LinkedList<>();
        Queue<Vertex<T>> vertexQ = new LinkedList<>();
        visited.add(start);
        vertexQ.add(start);
        while (vertexQ.size() > 0) {
            Vertex<T> v = vertexQ.remove();
            for (VertexDistance<T> vertexDist : adjList.get(v)) {
                if (!(visited.contains(vertexDist.getVertex()))) {
                    visited.add(vertexDist.getVertex());
                    vertexQ.add(vertexDist.getVertex());
                }
            }
        }
        return visited;
    }

    /**
     * Performs a depth first search (dfs) on the input graph, starting at
     * the parameterized starting vertex.
     *
     * When exploring a vertex, explore in the order of neighbors returned by
     * the adjacency list. Failure to do so may cause you to lose points.
     *
     * NOTE: This method should be implemented recursively. You may need to
     * create a helper method.
     *
     * You may import/use java.util.Set, java.util.List, and any classes that
     * implement the aforementioned interfaces, as long as they are efficient.
     *
     * The only instance of java.util.Map that you may use is the adjacency list
     * from graph. DO NOT create new instances of Map for DFS
     * (storing the adjacency list in a variable is fine).
     *
     * DO NOT modify the structure of the graph. The graph should be unmodified
     * after this method terminates.
     *
     * You may assume that the passed in start vertex and graph will not be null.
     * You may assume that the start vertex exists in the graph.
     *
     * @param <T>   The generic typing of the data.
     * @param start The vertex to begin the dfs on.
     * @param graph The graph to search through.
     * @return List of vertices in visited order.
     */
    public static <T> List<Vertex<T>> dfs(Vertex<T> start, Graph<T> graph) {
        // WRITE YOUR CODE HERE (DO NOT MODIFY METHOD HEADER)!
        Map<Vertex<T>, List<VertexDistance<T>>> adjList = graph.getAdjList();
        if (!(adjList.containsKey(start))) {
            throw new java.lang.IllegalArgumentException("Start must exist in the graph.");
        }
        List<Vertex<T>> visited = new LinkedList<>();
        visited = dfsHelper(graph, start, visited);
        return visited;
    }

    /**
     * This is the recursive helped method for DFS
     *
     * @param graph the graph to search through
     * @param start the vertex to begin the dfs on
     * @param visited list of vertices in visited order
     * @param <T> the generic typing of the data
     * @return list of vertices in visited order
     */
    private static <T> List<Vertex<T>> dfsHelper(Graph<T> graph, Vertex<T> start, List<Vertex<T>> visited) {
        visited.add(start);
        Map<Vertex<T>, List<VertexDistance<T>>> adjList = graph.getAdjList();
        for (VertexDistance<T> VertexDistance : adjList.get(start)) {
            if (!(visited.contains(VertexDistance.getVertex()))) {
                dfsHelper(graph, VertexDistance.getVertex(), visited);
            }
        }
        return visited;
    }       
}
