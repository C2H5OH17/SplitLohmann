class Params:
    def __init__(self):
        #FIXME:原参数
        # self.nominal_a = -0.14/0.08
        # self.C0 = 0.0193
        # self.lbda = 530e-09
        # self.f0 = 100e-03
        # self.fe = 40e-03
        # self.SLMpitch = 3.74e-06
        # self.W = 4.0
        # self.slmWidth = 4000
        # self.slmHeight = 2464

        # 修改后参数
        # 光学系统核心参数（保持不变）
        self.C0 = 0.0193  # 立方相位板参数（论文5.1节原型参数）
        self.lbda = 530e-09  # 波长（530nm，绿光）
        self.f0 = 100e-03  # 中继透镜焦距（100mm）
        self.fe = 40e-03  # 目镜焦距（40mm）
        self.W = 4.0  # 工作范围（4屈光度）

        # SLM硬件参数（根据新规格修改）
        self.SLMpitch = 8e-6  # SLM像素间距（8μm）
        self.slmWidth = 1920  # SLM宽度像素数（1920）
        self.slmHeight = 1080  # SLM高度像素数（1080）

        # 纵横比参数（根据SLM分辨率重新计算）
        self.nominal_a = -1920 / 1080  # 新SLM纵横比1920:1080=16:9，取负值表示坐标翻转

        # OLED硬件参数（新增）
        self.oledPitch = 8.1e-6  # OLED像素间距（8.1μm）
    def set_W(self, W):
        self.W = W
