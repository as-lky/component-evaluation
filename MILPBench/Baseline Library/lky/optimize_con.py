import optuna
study = optuna.create_study(study_name="MIKSmedium", direction="maximize", storage = "postgresql://luokeyun:lky883533600@localhost:5432/optuna_db")
