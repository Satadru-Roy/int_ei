from openmdao.main.api import VariableTree
from openmdao.lib.datatypes.api import Array, VarTree

from openmdao.lib.components.api import MetaModel 


class CustomMetaModel(MetaModel): 
    """Special metamodel with extra outputs. Currently hard coded to KrigingSurrogate""" 

    surrogate_data = VarTree(VariableTree(), iotype='out')

    def __init__(self, params=None, responses=None): 
        super(CustomMetaModel, self).__init__(params, responses)

        #create outputs for theta and R
        output_tree = self.get('surrogate_data')
        for name in responses:
            output_tree.add('%s_thetas' % name, Array())
            output_tree.add('%s_R' % name, Array())

    def execute(self): 
        update_R_theta = False
        if self._train: 
            update_R_theta = True

        super(CustomMetaModel, self).execute()

        if update_R_theta: 

            for name in self._surrogate_output_names:
                thetas_name = 'surrogate_data.%s_thetas' % name
                R_name = 'surrogate_data.%s_R' % name
                
                surrogate = self._get_surrogate(name)
                if surrogate is not None:
                    setattr(self, thetas_name, surrogate.thetas)
                    setattr(self, R_name, surrogate.R)
