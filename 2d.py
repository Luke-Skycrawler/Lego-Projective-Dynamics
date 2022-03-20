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
ti.init(arch=ti.x64)
ex = ti.Vector([1.,0.])
ey = ti.Vector([0.,1.])
linear_dict = {
    'x': ti.Vector.field(dim, dtype = float), # center
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
    return ti.Vector([c[1],-c[0]])

@ti.func
def R(r):
    return ti.Matrix.cols([r,inv(r)])
@ti.func
def signed_distance(i, I):  # i: index of brick; I: pos
    r,l,x = bricks[i].r, bricks[i].l, bricks[i].x
    Rm = R(r)
    p = I - x
    p = Rm @ p
    ret = 0.
    if 0 < p.x < (l-1) * diameter/res:
        ret = ti.abs(p.y)
    else : 
        ret = ti.min(p.norm(), (p - (l-1) * diameter/res * ex).norm())
    return ret

@ti.kernel
def init():
    for I in ti.grouped(index):
        index[I] = (I + 0.5) / grid
    bricks[0] = ti.Struct({
        'x': ti.Vector([0.2, 0.5]), # center
        'r': ti.Vector([1., 0.]), # rotation in complex number
        'l': 7,  # length
    })

@ti.kernel
def grid_sdf():
    for I in ti.grouped(sdf):
        sdf[I] = signed_distance(0,index[I])
        tot_sdf[None] += sdf[I]

def draw_all(ggui):
    to_np = lambda x: bricks.get_member_field(x).to_numpy()
    l,x,r = [to_np(i) for i in ['l','x','r']]
    ggui.lines(r,l,x, radius) 
    
@ti.kernel
def update(x:ti.ext_arr(), cnt: ti.i32):
    bricks[cnt].r = ti.Vector([x[0],x[1]]).normalized()
    # print(cnt, bricks[cnt].r)
# window = ti.ui.Window('Implicit Mass Spring System', res=(500, 500))
gui = ti.GUI("Vector Field", res=(res, res))
        
init()
with ti.Tape(tot_sdf):
    grid_sdf()
_grid = index.to_numpy().reshape((-1,2))
_grad = index.grad.to_numpy().reshape((-1,2)) / 40.0
# print(_grad)
cnt = 0
mxy = np.zeros((2,2),dtype = np.float32)
mcnt = 0
normalize = lambda x: x/np.linalg.norm(x)
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        print(e.key)
        if e.key == ti.GUI.LMB:
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) 
            mcnt = not mcnt
            print(mxy, mcnt,cnt)
            if mcnt :
                bricks[cnt] = ti.Struct({
                    'x': ti.Vector(mxy), # center
                    'r': ti.Vector([1., 0.]), # rotation in complex number
                    'l': 7,  # length
                })
            else:
                cnt += 1      
    elif mcnt :
        m = np.array(gui.get_cursor_pos())
        update(m-mxy,cnt)
        r,l,x = bricks[cnt].r, bricks[cnt].l, bricks[cnt].x
        gui.line(x, x + (l-1) * diameter/res * r, color = 0x888888, radius = radius)

    to_np = lambda x,y: bricks.get_member_field(x).to_numpy()[:y+1]
    l,x,r = [to_np(i,cnt-mcnt) for i in ['l','x','r']]
    if len(l):
        gui.lines(x, x + 6 * diameter/res * r, color = 0xFFFFFF, radius = radius)
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

