import torch
import numpy as np
import meshio
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
points=meshio.read("data/bunny_0.ply").points


all_points=np.zeros((600,points.shape[0],points.shape[1]))
for i in range(600):
    all_points[i]=meshio.read("data/bunny_"+str(i)+".ply").points

all_points=all_points.reshape(all_points.shape[0],-1)
np.save('points.npy',all_points)


data=np.load('points.npy')
pca=PCA(n_components=5)
pca.fit(data)

print("PCA loss is", np.linalg.norm(data-pca.inverse_transform(pca.transform(data)))/np.linalg.norm(data))
pca=KernelPCA(n_components=5,kernel='rbf',fit_inverse_transform=True)
pca.fit(data)
print("Data var", np.mean(np.var(data,axis=0)))
print("PCA Recon var",np.mean(np.var(pca.inverse_transform(pca.transform(data)),axis=0)))

print("KernelPCA loss is", np.linalg.norm(data-pca.inverse_transform(pca.transform(data)))/np.linalg.norm(data))
data=data.reshape(data.shape)
out_size=data.shape[1]
print("KernelPCA Recon var",np.mean(np.var(pca.inverse_transform(pca.transform(data)),axis=0)))
data=torch.tensor(data,dtype=torch.float32)


from revnet import RevNet
model=RevNet(5,out_size)



optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs=500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    recon=model(model.inverse(data)) 
    loss = torch.linalg.norm(recon-data)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        loss_tmp=torch.mean(torch.linalg.norm(model(model.inverse(data))-data,axis=1)/torch.linalg.norm(data,axis=1))
        print(loss_tmp)

print("Revnet loss is",loss_tmp)

model.eval()
with torch.no_grad():
    loss_tmp=torch.mean(torch.linalg.norm(model(model.inverse(data))-data,axis=1)/torch.linalg.norm(data,axis=1))
    print(loss_tmp)


    print(torch.mean(torch.var(model(model.inverse(data)),dim=0)).item())
    print(torch.mean(torch.var(data,dim=0)).item())
    latent_data=model.inverse(data)
    recon=model(model.inverse(data)) 
    for i in range(600):
        meshio.write("data/bunny_rec_"+str(i)+".ply", meshio.Mesh(points=recon[i].detach().numpy().reshape(-1,3),cells={}))

    latent_data=model.inverse(data).detach().numpy()
    np.save('latent_data.npy',latent_data)
