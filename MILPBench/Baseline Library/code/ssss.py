from pyscipopt import Model, SCIP_EVENTTYPE

def print_obj_value(model, event):
    print("!!!!!!!!!!")
#    print("New best solution found with objective value: {}".format(model.getObjVal()))

m = Model()
m.attachEventHandlerCallback(print_obj_value, [SCIP_EVENTTYPE.BESTSOLFOUND])
m.optimize()