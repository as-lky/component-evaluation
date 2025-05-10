import optuna
#study = optuna.create_study(study_name="MIKSmedium", direction="maximize", storage = "postgresql://luokeyun:lky883533600@localhost:5432/optuna_db")
#study = optuna.create_study(study_name="IShardNALNS", direction="maximize", storage = "postgresql://luokeyun:lky883533600@localhost:5432/IShard")
#study = optuna.create_study(study_name="MIKSCmediumNALNS", direction="maximize", storage = "postgresql://luokeyun:lky883533600@localhost:5432/MIKSCmedium")
#study = optuna.create_study(study_name="ISmediumNALNS", direction="maximize", storage = "postgresql://luokeyun:lky883533600@localhost:5432/ISmedium")
#study = optuna.create_study(study_name="SCmediumgurobiACP", direction="maximize", storage = "postgresql://luokeyun:lky883533600@localhost:5432/SCmedium")
#study = optuna.create_study(study_name="MIKSCmediumACP", direction="maximize", storage = "postgresql://luokeyun:lky883533600@localhost:5432/MIKSCmedium")
#SC bi_gcn_sr_ACP_
study = optuna.create_study(study_name="MVCmediumACP", direction="maximize", storage = "postgresql://luokeyun:lky883533600@localhost:5432/MVCmedium")
#MVC bi_gcn_nr_ACP_