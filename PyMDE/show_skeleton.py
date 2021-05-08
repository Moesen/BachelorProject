from pygel3d import graph, gl_display as gd

if __name__ == "__main__":
    g = graph.load("skeleton.graph")
    view = gd.Viewer()
    view.display(g)
    print(len(g.nodes()))