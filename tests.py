import taichi as ti
def Tests(globals):
    grid = globals.grid
    Diameter = globals.Diameter
    ex = globals.ex
    ey = globals.ey
    v_x = globals.v_x
    v_y = globals.v_y
    joints = globals.joints
    joints_lambda = globals.joints_lambda
    bricks = globals.bricks
    probes = globals.probes
    worldl = lambda l: (l-1) * Diameter
    @ti.kernel
    def init_case_3() -> ti.i32:    # for testing joints
        for I in ti.grouped(probes):
            probes[I] = (I + 0.5) / grid
        x2 = ti.Vector([0.1, 0.2])
        y2 = x2 + ey * worldl(7)
        
        x3 = x2 + ey * worldl(5)
        y3 = x3 + ex * worldl(7)
        
        x4 = y3 - ey * worldl(2)
        y4 = x4 + ey * worldl(7)
        bricks[0] = ti.Struct({
            'x': x2, 
            'y': y2,
            'l': 7,  # length
        })
        bricks[1] = ti.Struct({
            'x': x3, 
            'y': y3,
            'l': 7,  # length
        })
        bricks[2] = ti.Struct({
            'x': x4, 
            'y': y4,
            'l': 7,  # length
        })
        joints[0] = ti.Vector([0,1])
        joints[1] = ti.Vector([2,1])
        joints_lambda[0] = ti.Vector([1/3, 1.])
        joints_lambda[1] = ti.Vector([5/6, 0.])
        # print(f'jointed(0,1) = {jointed(0,1)}')
        v_x[0] = v_y[0] = ey * 5
        v_x[1] = v_y[1] = -ey * 5
        return 3

    @ti.kernel
    def init_case_4() -> ti.i32:    # for testing joints
        for I in ti.grouped(probes):
            probes[I] = (I + 0.5) / grid
        x2 = ti.Vector([0.1, 0.2])
        y2 = x2 + ey * worldl(7)
        
        x3 = x2 + ey * worldl(2)
        y3 = x3 + ex * worldl(7)
        
        e4 = ti.Vector([-0.6, +0.8])
        x4 = x3 + ex * worldl(4)
        y4 = x4 + e4 * worldl(7)
        bricks[0] = ti.Struct({
            'x': x2, 
            'y': y2,
            'l': 7,  # length
        })
        bricks[1] = ti.Struct({
            'x': x3, 
            'y': y3,
            'l': 7,  # length
        })
        bricks[2] = ti.Struct({
            'x': x4, 
            'y': y4,
            'l': 7,  # length
        })
        joints[0] = ti.Vector([0,1])
        joints[1] = ti.Vector([2,1])
        joints[2] = ti.Vector([0,2])
        joints_lambda[0] = ti.Vector([5/6, 1.])
        joints_lambda[1] = ti.Vector([1., 3/6])
        joints_lambda[2] = ti.Vector([1/6, 1/6])
        # print(f'jointed(0,1) = {jointed(0,1)}')
        # v_x[0] = v_y[0] = ey * 5
        # v_x[1] = v_y[1] = -ey * 5
        return 3

            
    @ti.kernel
    def init_case_2() -> ti.i32:    # for testing point-point contact
        for I in ti.grouped(probes):
            probes[I] = (I + 0.5) / grid
        x2 = ti.Vector([0.1, 0.2])
        y2 = x2 + ey * worldl(7)
        x3 = ti.Vector([0.5, y2.y + Diameter/4])
        y3 = x3 + ti.Vector([1., -0.5]).normalized() * worldl(7)
        bricks[0] = ti.Struct({
            'x': x3, 
            'y': y3,
            'l': 7,  # length
        })
        bricks[1] = ti.Struct({
            'x': x2, 
            'y': y2,
            'l': 7,  # length
        })
        v_x[0] = v_y[0] = -ex * 5
        v_x[1] = v_y[1] = ex * 5

        return 2

    @ti.kernel
    def init_case_1() -> ti.i32:    # for testing regular case
        for I in ti.grouped(probes):
            probes[I] = (I + 0.5) / grid
        x = ti.Vector([0.2, 0.05])
        y = x + ti.Vector([1., 0.]) * worldl(7)
        # x2 = ti.Vector([0.2 + worldl(7)/2, 0.2])
        x2 = ti.Vector([0.1, 0.5])
        # y2 = x2 + ti.Vector([1., 1.]).normalized() * worldl(7)
        y2 = x2 + ey * worldl(7)
        x3 = ti.Vector([0.5, (y2 + x2).y/2.])
        # y3 = x3 + ti.Vector([-1., 1.]).normalized() * worldl(7)
        y3 = x3 + ti.Vector([1., -0.5]).normalized() * worldl(7)
        # bricks[0] = ti.Struct({
        #     'x': x, 
        #     'y': y,
        #     'l': 7,  # length
        # })
        bricks[0] = ti.Struct({
            'x': x3, 
            'y': y3,
            'l': 7,  # length
        })
        bricks[1] = ti.Struct({
            'x': x2, 
            'y': y2,
            'l': 7,  # length
        })
        v_x[0] = v_y[0] = -ex * 5
        v_x[1] = v_y[1] = ex * 5

        return 2
    INIT_CASES = [init_case_1, init_case_2, init_case_3, init_case_4]
    N_JOINTS_LIST = [0, 0, 2, 3]

    return INIT_CASES, N_JOINTS_LIST
