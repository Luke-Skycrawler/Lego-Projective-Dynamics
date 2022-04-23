import taichi as ti
import numpy as np 
# (x|y|r)\[(.)\]    bricks[$2].$1

dim = 2
max_bricks = 1000
chunk = 1
res = 512
grid = 16
radius = 20
Radius = 18
diameter = 2 * radius
zero = -1e-9
Diameter = diameter / res
worldl = lambda l: (l-1) * Diameter
n_jacobian_iters = 40
allow_cross = False
flight = 5e-3
ti.init(arch=ti.x64)
ex = ti.Vector([1.,0.])
ey = ti.Vector([0.,1.])
linear_dict = {
    'x': ti.Vector.field(dim, dtype = float, needs_grad = True), # end
    'y': ti.Vector.field(dim, dtype = float, needs_grad = True), # end
    'r': ti.Vector.field(2, dtype = float, needs_grad = True), # rotation in complex number
    'l': ti.field(dtype = ti.i32, needs_grad= True),  # length
}
contacts = ti.field(dtype = ti.i32)
p_x = ti.Vector.field(dim, dtype = float)
p_y = ti.Vector.field(dim, dtype = float)
s_x = ti.Vector.field(dim, dtype = float)
s_y = ti.Vector.field(dim, dtype = float)

bricks = ti.StructField(linear_dict)
container = ti.root.pointer(ti.i, max_bricks)
container.place(bricks)
container.place(p_x, p_y, s_x, s_y, contacts)
probes = ti.Vector.field(2, float, (grid,grid))
sdf = ti.field(float, (grid, grid))
tot_sdf = ti.field(float, ())
tmp_dense_matrix = ti.field(ti.u8, (100,100))


ti.root.lazy_grad()


@ti.func
def inv(c):    # 1/ c    
    return ti.Vector([c[1],-c[0]])

@ti.func
def R(r):
    return ti.Matrix.cols([r,inv(r)])
@ti.func
def signed_distance(x_j, I):  # I: index of brick; x_j: pos
    l,x,y = bricks[I].l, bricks[I].x, bricks[I].y,
    Rm = R((y-x).normalized())
    p = x_j - x
    p = Rm @ p
    ret = 0.
    if 0 < p.x < worldl(l):
        ret = ti.abs(p.y)
    else : 
        ret = ti.min(p.norm(), (p - worldl(l) * ex).norm())
    return ret

@ti.kernel
def init():
    for I in ti.grouped(probes):
        probes[I] = (I + 0.5) / grid
    bricks[0] = ti.Struct({
        'x': ti.Vector([0.2, 0.5]), 
        'y': ti.Vector([0.,0.]),
        'r': ti.Vector([1., 0.]), # rotation in complex number
        'l': 7,  # length
    })

@ti.kernel
def grid_sdf(t: ti.i32):
    for I in ti.grouped(sdf):
        sdf[I] = signed_distance(probes[I], t)
        tot_sdf[None] += sdf[I]
@ti.kernel
def test_wrapper(cnt: ti.i32):
    for i in ti.grouped(probes):
        probes.grad[i] = nebla_sdf(probes[i], cnt)


# def draw_all(ggui):
#     to_np = lambda x: bricks.get_member_field(x).to_numpy()
#     l,x,r = [to_np(i) for i in ['l','x','r']]
#     ggui.lines(r,l,x, radius) 
    
@ti.kernel
def preview_brick(arr:ti.ext_arr(), cnt: ti.i32):
    r = bricks[cnt].r = ti.Vector([arr[0],arr[1]]).normalized()
    bricks[cnt].y = r * worldl(bricks[cnt].l) + bricks[cnt].x
    # print(cnt, bricks[cnt].r)

@ti.func
def update_fenwick(boolean, i, j): # collision at point x/y
    # pass
    tmp_dense_matrix[i,j] = boolean
    tmp_dense_matrix[j,i] = boolean

@ti.func
def contact(i,j):
    xc = signed_distance(bricks[i].x, j) < Diameter
    yc = signed_distance(bricks[i].y, j) < Diameter
    return xc or yc or contact_ev(i,j)

@ti.func
def contact_ev(i,j):
    return min(signed_distance(bricks[j].x, i), signed_distance(bricks[j].y, i)) < Diameter
    # return tmp_dense_matrix[i,j] & 2

@ti.kernel
def ve(i: ti.i32):
    for J in bricks:
        xc = signed_distance(bricks[i].x, J) < Diameter
        yc = signed_distance(bricks[i].y, J) < Diameter
        update_fenwick(xc or yc, i,J)

@ti.kernel
def ev(I:ti.i32):
    for j in bricks:
        t = min(signed_distance(bricks[j].x, I), signed_distance(bricks[j].y, I)) < Diameter
        # TODO: watch out the sdf 
        update_fenwick(t, j,I)
    
# @ti.kernel
# def _ee(i: ti.i32):
#     for j in bricks:
#         t = (bricks[j].r.transpose() @ bricks.x.grad[j]) * (bricks[j].r.transpose() @ bricks.y.grad[j]) < 0
#         update_fenwick(t[0], i,j)

# def sdf_gradient(I): # sdf gradient of brick I at ends of other bricks 
#     tot_sdf[None] = 0.
#     with ti.Tape(tot_sdf):
#         _grad_sdf(I)

# @ti.kernel
# def _grad_sdf(I:ti.i32):
#     for j in bricks:
#         tot_sdf[None] += signed_distance(bricks[j].x, I) + signed_distance(bricks[j].y, I)

@ti.kernel
def ee(i:ti.i32):
    for j in bricks:
        m11 = ti.Matrix.cols([bricks[j].x-bricks[i].x, bricks[i].r])
        m12 = ti.Matrix.cols([bricks[j].y-bricks[i].y, bricks[i].r])
        m21 = ti.Matrix.cols([bricks[j].x-bricks[i].x, bricks[j].r])
        m22 = ti.Matrix.cols([bricks[j].y-bricks[i].y, bricks[j].r])
        t = m11.determinant() * m12.determinant() < 0 and m21.determinant() * m22.determinant() < 0
        update_fenwick(t, i, j)

@ti.func
def nebla_sdf(x, J):
    r = (bricks[J].y - bricks[J].x).normalized()
    t =  R(r) @ (x - bricks[J].x)
    ret = ti.Vector.zero(float, 2)
    if 0 < t.x < worldl(7):
        ret = inv(r) * (1 if t.y > 0 else -1)
    elif t.x < 0:
        ret = (x - bricks[J].x).normalized()
    else:
        ret = (x - bricks[J].y).normalized()
    return ret
    
@ti.kernel
def copy_x_to_s():
    for i in bricks:
        s_x[i] = bricks[i].x
        s_y[i] = bricks[i].y

@ti.kernel
def project_local(cnt: ti.i32):   # local step for all bricks 
    
    for I in bricks:
        p_x[I] = p_y[I] = ti.Vector.zero(float, 2)
        contacts[I] = 1
        p_x[I] += (bricks[I].y - bricks[I].x) * (1-worldl(7)/(bricks[I].y-bricks[I].x).norm()) / 2
        p_y[I] -= (bricks[I].y - bricks[I].x) * (1-worldl(7)/(bricks[I].y-bricks[I].x).norm()) / 2
        for j in range(cnt):
            if j != I and contact(I,j):
                if contact_ev(I,j):
                    contact_point = bricks[j].x if signed_distance(bricks[j].x, I) < signed_distance(bricks[j].y, I) else bricks[j].y
                    p_x[I] -= (Diameter - signed_distance(contact_point, I)) * nebla_sdf(contact_point, I).normalized()
                    p_y[I] -= (Diameter - signed_distance(contact_point, I)) * nebla_sdf(contact_point, I).normalized()
                else:   # contact ve
                    p_x[I] += max(-signed_distance(bricks[I].x, j) + Diameter, 0.) * nebla_sdf(bricks[I].x, j).normalized()
                    p_y[I] += max(-signed_distance(bricks[I].y, j) + Diameter, 0.) * nebla_sdf(bricks[I].y, j).normalized()
                contacts[I] += 1


@ti.kernel
def minimize_global(cnt: ti.i32):
    for i in bricks:
        t1, t2 = bricks[i].x, bricks[i].y
        bricks[i].x += p_x[i]/ contacts[i]
        bricks[i].y += p_y[i]/ contacts[i]

        bricks[i].x *= contacts[i]/(contacts[i]+1)
        bricks[i].y *= contacts[i]/(contacts[i]+1)

        bricks[i].x += s_x[i] /(contacts[i] +1) 
        bricks[i].y += s_y[i] /(contacts[i] +1) 
        t3, t4 = bricks[i].x, bricks[i].y
        if i == cnt-1:
            print(f'residual :{(t3-t1).norm()}, {(t4-t2).norm()}, contacts: {contacts[i]}')
        
def solve_local_global(cnt): 
    copy_x_to_s()
    for i in range(n_jacobian_iters):
        print(f'iter{i}')
        project_local(cnt)
        minimize_global(cnt)
    
@ti.kernel
def collision_projection(i: ti.i32):
    for j in range(cnt):
        pass
        # if xc(j):

        
@ti.kernel
def check_fenwick(I:ti.i32, cnt: ti.i32) -> ti.i32:
    ret = 0
    for i in range(cnt):
        if i != I and tmp_dense_matrix[i,I]:
            ret = 1
    return ret

def probe_grid_sdf(cnt):
    # with ti.Tape(tot_sdf):
    #     grid_sdf(cnt)
    test_wrapper(cnt)
    _grid = probes.to_numpy().reshape((-1,2))
    _grad = probes.grad.to_numpy().reshape((-1,2)) / 40.0
    return _grid, _grad
    # print(_grad)

# window = ti.ui.Window('Implicit Mass Spring System', res=(500, 500))
gui = ti.GUI("Vector Field", res=(res, res))
        
init()
_grid, _grad = probe_grid_sdf(0)
cnt = 0
mxy = np.zeros((2,2),dtype = np.float32)
mcnt = 0
normalize = lambda x: x/np.linalg.norm(x)
t3 = False
while gui.running:
    
    to_np = lambda x,y: bricks.get_member_field(x).to_numpy()[:y+1]
    l,x,y,r = [to_np(i,cnt-mcnt) for i in ['l','x','y','r']]
    if len(l):
        gui.lines(x, y, color = 0x888888, radius = radius)
        gui.circles(np.vstack([x, y]), radius = Radius, color = 0x0)

    
    gui.circles(_grid, radius=3)
    gui.arrows(_grid, _grad, radius = 1)
    
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        print(e.key)
        if e.key == 'r':
            container.deactivate_all()
            cnt = 0
        if e.key == ti.GUI.LMB:
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) 
            mcnt = not mcnt if not (t3 and not allow_cross) else mcnt
            print(mxy, mcnt,cnt)
            if mcnt:
                bricks[cnt] = ti.Struct({
                    'x': ti.Vector(mxy), # center
                    'y': ti.Vector([0.,0.]), # center
                    'r': ti.Vector([1., 0.]), # rotation in complex number
                    'l': 7,  # length
                })
            elif not (t3 and not allow_cross):
                _grid, _grad = probe_grid_sdf(cnt)
                cnt += 1      
                solve_local_global(cnt)
                
    elif mcnt :
        m = np.array(gui.get_cursor_pos())
        preview_brick(m-mxy,cnt)
        tmp_dense_matrix.fill(0)
        ev(cnt)
        t1 = check_fenwick(cnt, cnt)
        ve(cnt)
        t2 = check_fenwick(cnt, cnt)
        ee(cnt)
        t3 = check_fenwick(cnt, cnt)
        
        color = 0x880000 if t3 else 0x440000 if t1 or t2 else  0x444444 
        r,l,x,y = bricks[cnt].r, bricks[cnt].l, bricks[cnt].x, bricks[cnt].y
        gui.line(x, y, color = color, radius = radius)
        gui.circle(x, color = 0x0, radius = Radius)
    gui.show()

# while window.running:
#     if window.get_event(ti.ui.PRESS):
#         if window.event.key == ti.ui.ESCAPE:
#             break
#     ti.ui.arrows(grid.to_numpy(), grid.grad.to_numpy(), radius = radius)
    # if window.is_pressed(ti.ui.SPACE):
    #     pause = not pause

    # if not pause:
    #     cloth.preview_brick(h)

    # canvas = window.get_canvas()
    # cloth.displayGGUI(canvas)
    # window.show()

