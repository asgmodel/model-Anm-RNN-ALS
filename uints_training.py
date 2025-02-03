# @title  train_model_wandb_long
import torch
import numpy as np
import wandb
# تدريب مدى طويل  ملاحظة :
def train_model_wandb_long(model,optimizer ,train_loader,test_loader, epochs=10,step_eval=10,name_project="anomaly-detection"):

    lst_lossall=[]
    lst_klall=[]
    lst_reconall=[]
    lst_anomaly_rate=[]
    lst_test_anomaly_rate=[]
    lst_test_mi=[]
    # model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_kl, epoch_recon = 0, 0, 0
        anomaly_rate = 0

        model.train()

        for i, x_batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = x_batch.to(device)
            batch_data = torch.clamp(batch_data , 0.0,0.99)
            total_loss, recon_loss, kl_divergence = model(batch_data)

            anomaly_rate += model.calculate_anomaly_rate(batch_data)

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_divergence.item()

        wandb.log({
            "epoch": epoch + 1,
            "total_loss": epoch_loss / len(train_loader),
            "reconstruction_loss": epoch_recon / len(train_loader),
            "kl_divergence": epoch_kl / len(train_loader),
            "anomaly_rate": anomaly_rate,
        })
        lst_lossall.append(epoch_loss/len(train_loader))
        lst_klall.append(epoch_kl/len(train_loader))
        lst_reconall.append(epoch_recon/len(train_loader))
        lst_anomaly_rate.append(anomaly_rate/len(train_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Recon Loss: {epoch_recon/len(train_loader):.4f}, KL: {epoch_kl/len(train_loader):.4f}, Anomaly Rate: {anomaly_rate/len(train_loader):.4f}")
        if (epoch+1) % step_eval == 0:
            avg_anomaly_rate,mu_all, actual_trajectories, generated_trajectories = test_model_wandb(model, test_loader)

            log_trajectories_to_wandb(actual_trajectories, generated_trajectories,num_samples=2)
            lst_test_anomaly_rate.append(avg_anomaly_rate)
            lst_test_mi.append(np.mean(mu_all))


    return model,{f"{model.state_type}":{"Loss":lst_lossall,
                                                 "Recon-Loss":lst_reconall,
                                                 "KL-Loss":lst_klall,
                                                 "anomaly_rate":lst_anomaly_rate,
                                                 "avg_anomaly_rate":lst_test_anomaly_rate,
                                                 "avg_mi":lst_test_mi,
                                                 }}
def test_model_wandb(model, test_loader):
    model.eval()
    anomaly_rates = []
    mu_all=[]
    actual_trajectories, generated_trajectories = [], []

    with torch.no_grad():
        for i, x_batch in enumerate(test_loader):
            batch_data =dataset[350+i].unsqueeze(0).to(device)
            batch_data = torch.clamp(batch_data , 0.0, 0.99)



            anomaly_rate = model.calculate_anomaly_rate(batch_data)
            anomaly_rates.append(anomaly_rate)
            mu_all.append(model.calc_mi(batch_data).cpu().numpy())

            actual_trajectories.append(batch_data.cpu().numpy())
            x_rect=model.get_logis(batch_data)


            x_rect=x_rect.cpu().numpy()
            generated_trajectories.append(x_rect)

    avg_anomaly_rate = np.mean(anomaly_rates)
    wandb.log({"average_anomaly_rate_test": avg_anomaly_rate})
    print(f"Average Anomaly Rate (Test): {avg_anomaly_rate:.4f}")
    wandb.log({"average_mi_test": np.mean(mu_all)})
    print(f"Average MI (Test): {np.mean(mu_all):.4f}")
    return avg_anomaly_rate,np.mean(mu_all), actual_trajectories, generated_trajectories

# def log_trajectories_to_wandb(actual, generated, num_samples=5):


#     for i in range(num_samples):
#         data = {
#             f"Actual Trajectory {i+1}": actual[i][0,:,0:2],
#             f"Generated Trajectory {i+1}": generated[i][0,:,0:2],
#         }
#         wandb.log(data)
def log_trajectories_to_wandb(actual, generated, num_samples=5):
    num_samples = min(num_samples, len(actual))
    for i in range(num_samples):
        # Prepare data for line plot

        actual_traj = actual[i][0]  # Extract X and Y for actual trajectory
        generated_traj = generated[i][0]  # Extract X and Y for generated trajectory
        actual_traj=dataset.dense_to_sparse(actual_traj)
        generated_traj=dataset.dense_to_sparse(generated_traj)
        actual_traj=actual_traj[:,0:2]
        generated_traj=generated_traj[:,0:2]


        # Convert to format suitable for W&B
        table = wandb.Table(data=[
            [int(x*LAT_BINS), int(y*LON_BINS), "Actual"] for x, y in actual_traj
        ] + [
            [int(x*LAT_BINS), int(y*LON_BINS), "Generated"] for x, y in generated_traj
        ], columns=["x", "y", "Type"])

        # Log the line plot
        wandb.log({f"Trajectory {i+1}": wandb.plot.line(
            table, "x", "y", "Type", title=f"Trajectory {i+1}",



        )})



# @title train_model_sciles

import wandb

# تدريب  شرائح
def train_model_sciles(model,optimizer ,train_loader,test_loader, epochs=10,step_eval=10,name_project="anomaly-detection"):

    # model.train()
    size_chack=10
    lst_lossall=[]
    lst_klall=[]
    lst_reconall=[]
    lst_anomaly_rate=[]
    lst_test_anomaly_rate=[]
    lst_test_mi=[]


    for epoch in range(epochs):
        epoch_loss, epoch_kl, epoch_recon = 0, 0, 0
        anomaly_rate = 0
        model.train()


        for i, x_batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = x_batch.to(device)
            batch_data = torch.clamp(batch_data , 0.0, 0.99)

            chacks=[ chack for chack in range(size_chack)]
            sub_loss,sub_recon_loss, sub_kl_divergence = 0, 0, 0
            for chack in chacks:
                sub_batch=batch_data[:,chack:chack+size_chack,:]

                optimizer.zero_grad()

                total_loss, recon_loss, kl_divergence = model(sub_batch)




                total_loss.backward()


                optimizer.step()

                sub_loss += total_loss.item()
                sub_recon_loss += recon_loss.item()
                sub_kl_divergence += kl_divergence.item()


            epoch_loss += sub_loss/len(chacks)
            epoch_recon += sub_recon_loss/len(chacks)
            epoch_kl += sub_kl_divergence/len(chacks)
            anomaly_rate += model.calculate_anomaly_rate(batch_data)


        wandb.log({
            "epoch": epoch + 1,
            "total_loss": epoch_loss / len(train_loader),
            "reconstruction_loss": epoch_recon / len(train_loader),
            "kl_divergence": epoch_kl / len(train_loader),
            "anomaly_rate": anomaly_rate,
        })
        lst_lossall.append(epoch_loss/len(train_loader))
        lst_klall.append(epoch_kl/len(train_loader))
        lst_reconall.append(epoch_recon/len(train_loader))
        lst_anomaly_rate.append(anomaly_rate/len(train_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Recon Loss: {epoch_recon/len(train_loader):.4f}, KL: {epoch_kl/len(train_loader):.4f}, Anomaly Rate: {anomaly_rate/len(train_loader):.4f}")
        if (epoch+1) % step_eval == 0:
            avg_anomaly_rate,mu_all, actual_trajectories, generated_trajectories = test_model_wandb(model, test_loader)

            log_trajectories_to_wandb(actual_trajectories, generated_trajectories,num_samples=2)
            lst_test_anomaly_rate.append(avg_anomaly_rate)
            lst_test_mi.append(np.mean(mu_all))


    return model,{f"{model.state_type}":{"Loss":lst_lossall,
                                                 "Recon-Loss":lst_reconall,
                                                 "KL-Loss":lst_klall,
                                                 "anomaly_rate":lst_anomaly_rate,
                                                 "avg_anomaly_rate":lst_test_anomaly_rate,
                                                 "avg_mi":lst_test_mi,
                                                 }}
def test_model_wandb(model, test_loader):
    model.eval()
    anomaly_rates = []
    mu_all=[]
    actual_trajectories, generated_trajectories = [], []

    with torch.no_grad():
        for i, x_batch in enumerate(test_loader):
            batch_data =dataset[350+i].unsqueeze(0).to(device)
            batch_data = torch.clamp(batch_data , 0.0, 0.99)



            anomaly_rate = model.calculate_anomaly_rate(batch_data)
            anomaly_rates.append(anomaly_rate)
            mu_all.append(model.calc_mi(batch_data).cpu().numpy())

            actual_trajectories.append(batch_data.cpu().numpy())
            x_rect=model.get_logis(batch_data)


            x_rect=x_rect.cpu().numpy()
            generated_trajectories.append(x_rect)

    avg_anomaly_rate = np.mean(anomaly_rates)
    wandb.log({"average_anomaly_rate_test": avg_anomaly_rate})
    print(f"Average Anomaly Rate (Test): {avg_anomaly_rate:.4f}")
    wandb.log({"average_mi_test": np.mean(mu_all)})
    print(f"Average MI (Test): {np.mean(mu_all):.4f}")
    return avg_anomaly_rate,np.mean(mu_all), actual_trajectories, generated_trajectories

# def log_trajectories_to_wandb(actual, generated, num_samples=5):


#     for i in range(num_samples):
#         data = {
#             f"Actual Trajectory Four-hot  {i+1}": actual[i][0,:,0:2],
#             f"Generated Trajectory Four-hot {i+1}": generated[i][0,:,0:2],
#         }
#         wandb.log(data)
def log_trajectories_to_wandb(actual, generated, num_samples=5):

    num_samples = min(num_samples, len(actual))
    for i in range(num_samples):
        actual_traj = actual[i][0]
        generated_traj = generated[i][0]
        actual_traj=dataset.dense_to_sparse(actual_traj)
        generated_traj=dataset.dense_to_sparse(generated_traj)
        actual_traj=actual_traj[:,0:2]
        generated_traj=generated_traj[:,0:2]
        table = wandb.Table(data=[
            [int(x*LAT_BINS), int(y*LON_BINS), "Actual"] for x, y in actual_traj
        ] + [
            [int(x*LAT_BINS), int(y*LON_BINS), "Generated"] for x, y in generated_traj
        ], columns=["x", "y", "Type"])

        wandb.log({f"Trajectory {i+1}": wandb.plot.line(
            table, "x", "y", "Type", title=f"Trajectory {i+1}",



        )})



