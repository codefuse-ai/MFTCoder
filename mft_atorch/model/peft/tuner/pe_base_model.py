class PEBaseModel:
    """PEtuning的基类模型，定义了PEtuning模型都该有的方法"""

    def __init__():
        return

    def get_model(self):
        """对模型进行修改，冻结参数或者插入可训模块"""
        pass

    @classmethod
    def restore(self, model=None, path=None):
        """从path恢复PE模型

        Args:
            model (_type_, optional): 原始模型. Defaults to None.
            path (_type_, optional): 增量路径. Defaults to None.
        """
        pass
