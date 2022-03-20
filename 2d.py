from matplotlib import container
import taichi as ti
import numpy as np 


dim = 2
max_bricks = 1000
chunk = 1
res = 512
grid = 16
radius = 20
diameter = 2 * radius
ti.init(arch=ti.cuda)
ex = ti.Vector([1.,0.])
ey = ti.Vector([0.,1.])
linear_dict = {
    'x': ti.Vector.field(dim, dtype = float), # center
    'R': ti.Matrix.field(dim,dim, dtype = float), # rotation matrix
    'r': ti.Vector.field(2, dtype = float), # rotation in complex number
    'l': ti.field(dtype = ti.i32),  # length
}
bricks = ti.StructField(linear_dict)
container = ti.root.pointer(ti.i, max_bricks)
container.place(bricks)
index = ti.Vector.field(2, float, (grid,grid))
sdf = ti.field(float, (grid, grid))
tot_sdf = ti.field(float, ())
ti.root.lazy_grad()

@ti.func
def inv(c):    # 1/ c
    
    # return ti.Vector([c[0], -c[1]]) / c.norm_sqr()   
    return ti.Vector([c[1],-c[0]])


@ti.kernel
def init():
    for I in ti.grouped(index):
        index[I] = (I + 0.5) / grid
    bricks[0] = ti.Struct({
        'x': ti.Vector([0.2, 0.5]), # center
        'R': ti.Matrix.zero(float, 2,2), # rotation matrix
        'r': ti.Vector([1., 0.]), # rotation in complex number
        'l': 7,  # length
    })
    t = bricks[0].r
    bricks[0].R = ti.Matrix.cols([t,inv(t)])

@ti.kernel
def calc_sdf(t: ti.template(),l: ti.template(),m: ti.template(),x: ti.template()):
    for I in ti.grouped(sdf):
        p = index[I] - x
        p = m @ p
        if 0 < p.x < (l-1) * diameter/res:
            sdf[I] = ti.abs(p.y)
        else : 
            sdf[I] = ti.min(p.norm(), (p - (l-1) * diameter/res * ex).norm())
        tot_sdf[None] += sdf[I]

def draw_all(ggui):
    to_np = lambda x: bricks.get_member_field(x).to_numpy()
    l,x,r = [to_np(i) for i in ['l','x','r']]
    ggui.lines(r,l,x, radius) 
    

# window = ti.ui.Window('Implicit Mass Spring System', res=(500, 500))
gui = ti.GUI("Vector Field", res=(res, res))


init()
r,l,R,x = bricks[0].r, bricks[0].l, bricks[0].R, bricks[0].x
with ti.Tape(tot_sdf):
    calc_sdf(r,l,R,x)
_grid = index.to_numpy().reshape((-1,2))
_grad = index.grad.to_numpy().reshape((-1,2)) / 40.0
print(_grad, r,l, R, x)
for k in range(1000000):
    gui.line(x, x + (l-1) * diameter/res * ex, color = 0xFFFFFF, radius = radius)
    gui.circles(_grid, radius=3)
    gui.arrows(_grid, _grad, radius = 1)
    gui.show()

# while window.running:
#     if window.get_event(ti.ui.PRESS):
#         if window.event.key == ti.ui.ESCAPE:
#             break
#     ti.ui.arrows(grid.to_numpy(), grid.grad.to_numpy(), radius = radius)
    # if window.is_pressed(ti.ui.SPACE):
    #     pause = not pause

    # if not pause:
    #     cloth.update(h)

    # canvas = window.get_canvas()
    # cloth.displayGGUI(canvas)
    # window.show()

