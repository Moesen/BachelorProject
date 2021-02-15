from pygel3d import hmesh, gl_display, graph

m = hmesh.obj_load(" ../../GEL/data/bunny.obj")
v = gl_display.Viewer()
v.display(m)

hmesh.quadric_simplify(m, 0, 1)
v.display(m)

g = graph.from_mesh(m)
v.display(m, g, mode="x")

s = graph.LS_skeleton(g)

v.display(m, s, mode="x")
v.display(m, s, mode="w")
hmesh.close_holes(m)

hmesh.triangulate(m)

v.display(m, s, mode="x")

hmesh.quadric_simplify(m, 0.25)
v.display(m, s, mode="w")
g = graph.from_mesh(m)

s = graph.LS_skeleton(g)

v.display(m, s, mode="x")