from pygel3d import hmesh, gl_display

m = hmesh.obj_load(" ../../GEL/data/bunny.obj")
v = gl_display.Viewer()
v.display(m)