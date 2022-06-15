model_setup = {
    'transformer'       : {
        'nhead':4,
        'd_hid':400,
        'd_model' : 400,
        'nlayers':4,
    },
    'decoder':{
        'output_seq':True
    },
    'critic':{
        'd_output' : 1,
        'output_seq':False
    },
}