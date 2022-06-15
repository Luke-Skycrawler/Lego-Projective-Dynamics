import taichi as ti
import numpy as np
# (x|y|r)\[(.)\]    bricks[$2].$1

dim = 2
max_bricks, max_joints = 1000, 500
chunk = 1
res = 512
grid = 16
radius = 20
Radius = 18
diameter = 2 * radius
zero = -1e-9
Diameter = diameter / res
n_jacobian_iters = 100
n2_jacobian_iters = 1
epsilon = 1. + 0.
dt = 5e-4
debug = False
debug_v = False
allow_cross = False
flight = 5e-3
worldl = lambda l: (l-1) * Diameter
ti.init(arch=ti.x64)
ex = ti.Vector([1.,0.])
ey = ti.Vector([0.,1.])
linear_dict = {
    'x': ti.Vector.field(dim, dtype = float, needs_grad = True), # end
    'y': ti.Vector.field(dim, dtype = float, needs_grad = True), # end
    'l': ti.field(dtype = ti.i32, needs_grad= True),  # length
}
contacts = ti.field(dtype = ti.i32)
p_x = ti.Vector.field(dim, dtype = float)
p_y = ti.Vector.field(dim, dtype = float)
p_vx = ti.Vector.field(dim, dtype = float)
p_vy = ti.Vector.field(dim, dtype = float)
s_x = ti.Vector.field(dim, dtype = float)
s_y = ti.Vector.field(dim, dtype = float)
v_x = ti.Vector.field(dim, dtype = float)
v_y = ti.Vector.field(dim, dtype = float)
q_x = ti.Vector.field(dim, dtype = float)
q_y = ti.Vector.field(dim, dtype = float)
q_vx = ti.Vector.field(dim, dtype = float)
q_vy = ti.Vector.field(dim, dtype = float)
joints = ti.Vector.field(2, dtype = ti.i32)
joints_lambda = ti.Vector.field(2, dtype = float)
joints_container = ti.root.pointer(ti.i, 10).pointer(ti.i, 10).pointer(ti.i, max_joints // 100)
joints_container.place(joints, joints_lambda)
bricks = ti.StructField(linear_dict)
container = ti.root.pointer(ti.i, 10).pointer(ti.i, 10).pointer(ti.i, max_joints // 100)
container.place(bricks)
container.place(p_x, p_y,p_vx, p_vy, s_x, s_y, v_x, v_y, q_x, q_y, q_vx, q_vy, contacts)
probes = ti.Vector.field(2, float, (grid,grid))
sdf = ti.field(float, (grid, grid))
tot_sdf = ti.field(float, ())
tmp_dense_matrix = ti.field(ti.u8, (100,100))
# temporary joint info 
joints_preview_info = ti.Vector.field(2, dtype = ti.i32, shape = ())
completed_joint_number = ti.field(dtype = ti.i32, shape = ())

lambda_preview_info = ti.Vector.field(2, dtype = float, shape = ())
jn = ti.field(dtype = ti.i32)
container.place(jn)

ti.root.lazy_grad()

class Globals():
    def __init__(self):
        self.dim = dim
        self.max_bricks = max_bricks
        self.max_joints = max_joints
        self.chunk = chunk
        self.res = res
        self.grid = grid
        self.radius = radius
        self.Radius = Radius
        self.diameter = diameter
        self.zero = zero
        self.Diameter = Diameter
        self.n_jacobian_iters = n_jacobian_iters
        self.n2_jacobian_iters = n2_jacobian_iters
        self.epsilon = epsilon
        self.dt = dt
        self.debug = debug
        self.debug_v = debug_v
        self.allow_cross = allow_cross
        self.flight = flight
        self.ex = ex
        self.ey = ey
        self.contacts = contacts
        self.p_x = p_x
        self.p_y = p_y
        self.p_vx = p_vx
        self.p_vy = p_vy
        self.s_x = s_x
        self.s_y = s_y
        self.v_x = v_x
        self.v_y = v_y
        self.q_x = q_x
        self.q_y = q_y
        self.q_vx = q_vx
        self.q_vy = q_vy
        self.joints = joints
        self.joints_lambda = joints_lambda
        self.bricks = bricks
        self.probes = probes
        self.sdf = sdf
        self.tot_sdf = tot_sdf
        self.tmp_dense_matrix = tmp_dense_matrix
        self.joints_preview_info = joints_preview_info
        self.completed_joint_number = completed_joint_number
        self.lambda_preview_info = lambda_preview_info
        self.jn = jn

globals = Globals()
from utils import Utils
from bricks import Bricks

inv, R, signed_distance, grid_sdf, test_wrapper, update_fenwick, contact, contact_ev, ve, ev, _ee, ee, nebla_sdf, copy_x_to_s, probe_grid_sdf, contact_point_on_edge, jointed = Utils(globals)
project_v, project_local, preview_brick = Bricks(globals, signed_distance, inv, contact_ev, contact, jointed, nebla_sdf, contact_point_on_edge)


@ti.func
def mingle(i, lam):
    return bricks[i].x * lam + bricks[i].y * (1- lam)

@ti.func
def interpolate(x, i):
    n = bricks[i].l - 1.
    lam = contact_point_on_edge(i, x)
    _lam = ti.cast(ti.round(lam * n), float) / n
    return bricks[i].x * _lam + bricks[i].y * (1-_lam), _lam

@ti.kernel
def preview_neighbouring_joint(arr: ti.ext_arr(), ret: ti.ext_arr()) -> ti.i32:
    x = ti.Vector([arr[0], arr[1]])
    hole = ti.Vector.zero(float, 2)
    b = 0
    for i in bricks:
        if signed_distance(x, i) < Diameter /2:
            hole, lam = interpolate(x, i)
            ret[0], ret[1] = i, lam
            b = 1
    arr[0], arr[1] = hole.x, hole.y
    return b

@ti.kernel
def add_joint(cnt: ti.i32, arr: ti.ext_arr()):
    j = ti.Vector([ti.cast(arr[0], ti.i32), -1])
    jl = ti.Vector([arr[1], -1.])
    # joints_pos[cnt] = x
    joints[cnt], joints_lambda[cnt] = j, jl
    
    # joints[cnt] = ti.Vector([i, j])
    # joints_lambda[cnt] = ti.Vector([l1, l2])

@ti.kernel
def get_joints_pos(arr: ti.ext_arr()):
    for j in joints:
        x = mingle(joints[j].x, joints_lambda[j].x)
        arr[j, 0], arr[j, 1] = x.x, x.y


@ti.kernel
def complete_joint():
    j = completed_joint_number[None]
    joints[j] = joints_preview_info[None]
    joints_lambda[j] = lambda_preview_info[None]

@ti.kernel
def solve_joints():
    for j in joints:
        if joints[j].y != -1:
            i1, i2 = joints[j].x, joints[j].y
            x1 = mingle(i1, joints_lambda[j].x)
            x2 = mingle(i2, joints_lambda[j].y)
            p_x[i1] += x2 - x1
            p_y[i1] += x2 - x1
            p_x[i2] += x1 - x2
            p_y[i2] += x1 - x2
            contacts[i1] += 1
            contacts[i2] += 1

@ti.kernel
def global_v(cnt: ti.i32):
    for i in bricks:
        r1 = (bricks[i].y - bricks[i].x).normalized()
        px0 = q_vx[i] + (q_vy[i] - q_vx[i]).dot(r1) * r1 /2
        py0 = q_vy[i] - (q_vy[i] - q_vx[i]).dot(r1) * r1 /2
        # p_vx[i] += px0, p_vy[i] = px0, py0
        q_vx[i] = p_vx[i] / max(contacts[i] - 1, 1) + px0
        q_vy[i] = p_vy[i] / max(contacts[i] - 1, 1) + py0


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
        if i == cnt-1 and debug:
            print(f'residual :{(t3-t1).norm()}, {(t4-t2).norm()}, contacts: {contacts[i]}')
        
def solve_local_global(cnt, n_joints): 
    copy_x_to_s(s_x, s_y)
    for i in range(n_jacobian_iters):
        if debug:
            print(f'iter{i}')
        project_local(cnt, n_joints)
        solve_joints()
        minimize_global(cnt)

def solve_v(cnt, n_joints):
    q_vx.copy_from(v_x)
    q_vy.copy_from(v_y)
    for i in range(n2_jacobian_iters):
        project_v(cnt, n_joints, v_x, v_y, q_vx, q_vy)
        global_v(cnt)
    v_x.copy_from(q_vx)
    v_y.copy_from(q_vy)
        
@ti.kernel
def check_fenwick(I:ti.i32, cnt: ti.i32) -> ti.i32:
    ret = 0
    for i in range(cnt):
        if i != I and tmp_dense_matrix[i,I]:
            ret = 1
    return ret

@ti.kernel
def add_gravity(g: ti.i32):
    for i in bricks:
        bricks[i].x += v_x[i] * dt
        bricks[i].y += v_y[i] * dt
        _g = -ey if g == 1 else ex if g == 2 else -ex if g == 3 else ti.Vector.zero(float, 2)
        v_x[i] += dt * 500. * _g 
        v_y[i] += dt * 500. * _g

        
@ti.kernel
def update_velocity():
    for i in bricks:
        v_x[i] = (bricks[i].x - q_x[i]) / dt
        v_y[i] = (bricks[i].y - q_y[i]) / dt


@ti.kernel
def preview_possible_assemble(m: ti.ext_arr(), mxy: ti.ext_arr(), ret: ti.ext_arr(), cnt: ti.i32)->ti.i32:
    mouse = ti.Vector([m[0], m[1]])
    bx = ti.Vector([mxy[0], mxy[1]])
    b = 0
    for j in joints:
        if joints[j].y == -1 and b == 0:
            x = mingle(joints[j].x, joints_lambda[j].x)
            if (x-mouse).norm() < Diameter/2:
                b += 1
                ret[0], ret[1] = x.x, x.y


                r = (x - bx).normalized()
                bricks[cnt].y = r * worldl(bricks[cnt].l) + bricks[cnt].x

                completed_joint_number[None] = j
                joints_preview_info[None] = joints[j]
                lambda_preview_info[None] = joints_lambda[j]


                joints_preview_info[None].y = cnt
                hole, lam = interpolate(x, cnt)
                lambda_preview_info[None].y = lam
                ''' FIXME: 
                1. adjust position along the brick; 
                2. if e-v contact after the move, warning color
                3. invalidate e-e contact with other bricks
                4. global solve after confirming the assemble

                '''
    return b

# window = ti.ui.Window('Implicit Mass Spring System', res=(500, 500))
gui = ti.GUI("LEGO master breaker", res=(res, res))

from tests import Tests

CASE = 2
INIT_CASES, N_JOINTS_LIST = Tests(globals)
init, n_joints = INIT_CASES[CASE], N_JOINTS_LIST[CASE]

cnt = init()
_grid, _grad = probe_grid_sdf(0)
mxy = np.zeros((2,2),dtype = np.float32)
mcnt = 0
normalize = lambda x: x/np.linalg.norm(x)
t3 = False
pause = False
gravity = False
to_np = lambda x,y: bricks.get_member_field(x).to_numpy()[:y+1]
BRICKS_MODE = 0
JOINT_MODE = 1
mode = BRICKS_MODE
tmp_joint = np.zeros((2,), dtype = np.float32)
new_joint_allowed = False
while gui.running:
    
    l,x,y = [to_np(i,cnt-mcnt) for i in ['l','x','y']]
    if len(l):
        gui.lines(x, y, color = 0x888888, radius = radius)
        gui.circles(np.vstack([x, y]), radius = Radius, color = 0x0)
        v = np.vstack([v_x.to_numpy()[:cnt-mcnt + 1], v_y.to_numpy()[:cnt-mcnt + 1]]) / 100.0
        gui.arrows(np.vstack([x, y]), v, radius = 3, color= 0x888800)
    
    gui.circles(_grid, radius=3)
    gui.arrows(_grid, _grad, radius = 1)
    if not pause:
        copy_x_to_s(q_x,q_y)
        add_gravity(gravity)
        solve_local_global(cnt, n_joints)
        solve_v(cnt, n_joints)
        # update_velocity()
    if gui.get_event(ti.GUI.PRESS):
        e = gui.event
        print(e.key)
        if e.key == 'r' or e.key =='c':
            container.deactivate_all()
            joints_container.deactivate_all()
            cnt = init() if e.key == 'r' else 0
            n_joints = N_JOINTS_LIST[CASE]
        elif e.key == 's':
            pause = not pause
        elif e.key == 'g':
            gravity = not gravity
        elif e.key == ti.GUI.LEFT:
            gravity = 3
        elif e.key == ti.GUI.RIGHT:
            gravity = 2
        elif e.key == ti.GUI.DOWN:
            gravity = 1
        elif e.key == 'f':
            mode = not mode
        elif e.key == ti.GUI.LMB:
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) 
            if mode == BRICKS_MODE:    
                mcnt = not mcnt if not (t3 and not allow_cross) else mcnt
                print(mxy, mcnt,cnt)
                if mcnt:
                    bricks[cnt] = ti.Struct({
                        'x': ti.Vector(mxy), # center
                        'y': ti.Vector(mxy), # center
                        'l': 7,  # length
                    })
                elif not (t3 and not allow_cross):
                    if picked:
                        complete_joint()
                    _grid, _grad = probe_grid_sdf(cnt)
                    cnt += 1      
                    solve_local_global(cnt)
            elif mode == JOINT_MODE and new_joint_allowed: # joint mode
                add_joint(n_joints, tmp_joint)
                n_joints += 1
                # print(n_joints)
                # print(joints.to_numpy()[:n_joints], joints_lambda.to_numpy()[:n_joints])
                
    elif mcnt or mode == JOINT_MODE:
        m = np.array(gui.get_cursor_pos())
        if mcnt:
            preview_brick(m-mxy,cnt)
            tmp_dense_matrix.fill(0)
            # tj = ej(cnt)
            picked = preview_possible_assemble(m, mxy, tmp_joint, cnt)
            ev(cnt)
            t1 = check_fenwick(cnt, cnt)
            ve(cnt)
            t2 = check_fenwick(cnt, cnt)
            ee(cnt)
            t3 = check_fenwick(cnt, cnt) and not picked
            
            color = 0x880000 if t3 else 0x440000 if (t1 or t2 and not picked) else 0x444444 
            l,x,y = bricks[cnt].l, bricks[cnt].x, bricks[cnt].y
            gui.line(x, y, color = color, radius = radius)
            gui.circle(x, color = 0x0, radius = Radius)
            if picked:
                gui.circle(tmp_joint, color = 0xffffff, radius = Radius * 0.9)
        else:
            new_joint_allowed = preview_neighbouring_joint(m, tmp_joint)
            # tmp_joint = m
            gui.circle(m, color = 0xffffff, radius = Radius * 0.8)
        
    if n_joints > 0:
        jx = np.zeros((n_joints, 2), np.float32)
        get_joints_pos(jx)
        gui.circles(jx, color = 0xffffff, radius = Radius * 0.8)
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

