#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch as torch
import torch.nn as nn
import  torch.nn.functional as F
from config import config_Amazon as cfg
device = torch.device('cpu')
# device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class HGRec(nn.Module):
    def __init__(self,
                 cfg,
                 U_adj_list=None,  # include UI
                 I_adj_list=None,  # include IU
                 UIUI=None,  # UIUI矩阵比较特殊,单独传进来
                 have_feature=None  # [u_fea, i_fea]
                 ):
        super(HGRec,self).__init__()
        self.cfg = cfg
        self.gnn_type = cfg.gnn_type
        self.depth = cfg.depth  # depth for all aggregator
        self.U_MP = cfg.U_MP
        self.I_MP = cfg.I_MP
        self.u_gnn_depth = cfg.u_gnn_depth
        self.i_gnn_depth = cfg.i_gnn_depth

        # 交互方式
        self.interaction_type = cfg.interaction_type
        self.co_attention_transform = cfg.co_attention_transform
        self.co_attention_type = cfg.co_attention_type

        # hyper para
        self.emb_dim = cfg.emb_dim
        self.l2_reg = cfg.l2_reg

        self.init_lr = cfg.lr
        self.lr = cfg.lr
        self.max_degree = cfg.max_degree
        self.drop = torch.nn.Dropout(0.2)
        self.U_adj_list = U_adj_list
        self.I_adj_list = I_adj_list
        self.UIUI = UIUI
        # fea
        self.use_fea = cfg.use_fea
        self.user_fea_embed = cfg.user_fea_embed
        self.have_feature = have_feature  # 2*3,3*5

        if self.have_feature:
            self.num_u_fea = have_feature[0].shape[1]
            self.num_i_fea = have_feature[1].shape[1]

        # multi_embed_1.get_shape().as_list()[1]
        self.num_mp_1 = len(self.U_MP)  # = len(U_adj_list) + 1  # if cfg.if_UIUI else len(U_adj_list)
        self.num_mp_2 = len(self.I_MP)  # = len(I_adj_list) + 1  # if cfg.if_UIUI else len(I_adj_list)
        self.coef_drop = cfg.coef_drop
        self.ffd_drop = cfg.ffd_drop

        if self.have_feature and self.use_fea:
            self.num_dim_1 = have_feature[0].shape[1]
            self.num_dim_2 = have_feature[1].shape[1]
        else:
            self.num_dim_1 = self.emb_dim
            self.num_dim_2 = self.emb_dim
        #         self.n_users = 3  # U_adj_list[0].shape[0]
        #         self.n_items = 5 # I_adj_list[0].shape[0]
        self.n_users = U_adj_list[0].shape[0] # 943
        self.n_items =  I_adj_list[0].shape[0] #1682
        hidden_size = self.emb_dim * (self.depth + 1)
        self.multi_user_embed, self.multi_item_embed = None,None
        self.layer_trans_u = nn.Linear(hidden_size,self.emb_dim,bias=False).to(device)
        self.layer_trans_i = nn.Linear(hidden_size,self.emb_dim,bias=False).to(device)
        print(f"Model: HGRec including UI, (U-mp: {cfg.U_MP}, M-mp: {cfg.I_MP} "
              f"Agg: {self.gnn_type},  Interaction: {self.interaction_type})")

        self.ngcf_Linear1 = nn.Linear(self.emb_dim, self.emb_dim, bias=False).to(device)
        self.ngcf_Linear2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False).to(device)
        self.item_i_omega = torch.rand(size=[self.emb_dim])
        self.user_u_omega = torch.rand(size=[self.emb_dim])

        if self.interaction_type == 'han':

            attention_size = 64
            hidden_size_item = attention_size *(self.depth + 1)
            hidden_size_user = attention_size *(self.depth + 1)
            self.item_Linear = nn.Linear(hidden_size_item, attention_size,
                                         bias=True,device=device
                                         )
            self.user_Linear = nn.Linear(hidden_size_user, attention_size,
                                         bias=True,device=device
                                         )
        ######################################################################################
        if not self.use_fea:
            print('init user/item embedding via xavier initialization')

            self.init_user_embed = torch.empty(
                self.n_users, self.num_dim_1, requires_grad=True,dtype=torch.half).to(device)
            nn.init.xavier_normal_(self.init_user_embed)
            self.init_item_embed = torch.empty(self.n_items, self.num_dim_1, requires_grad=True,dtype=torch.half).to(device)
            nn.init.xavier_normal_(self.init_item_embed)

        if self.have_feature and self.use_fea:
            if not self.user_fea_embed:
                print('init user/item embedding via loading their features ')
                self.init_user_embed = torch.DoubleTensor(size=(self.n_users, self.num_dim_1))
                self.init_item_embed = torch.DoubleTensor(self.n_items, self.num_dim_2)
                self.init_user_embed = nn.init.xavier_normal_(self.init_user_embed)
                self.init_item_embed = nn.init.xavier_normal_(self.init_user_embed)
            else:
                print('init user/item embedding via averaged fea embedding ')
                # self. u_fea_embed = tf.Variable(initializer(
                #     [self.num_u_fea, self.embed_dim]), name='user_fea_embedding')
                # self.i_fea_embed = tf.Variable(initializer(
                #     [self.num_i_fea, self.embed_dim]), name='item_fea_embedding')
                # self.init_user_embed = tf.matmul(self.have_feature[0], self.u_fea_embed)
                # self.init_item_embed = tf.matmul(self.have_feature[1], self.i_fea_embed)
        print(
            f"init user: {self.init_user_embed.shape}, "
            f"init Item: {self.init_item_embed.shape}")  # [10,2][20,3]
        self.A = torch.zeros([cfg.emb_dim,cfg.emb_dim], requires_grad=True).to(device)
        self.A = nn.init.xavier_normal_(self.A)

        # place hoders

    def forward(self,batch_data):
        if self.training:
            # if self.multi_item_embed == None:
            self.multi_user_embed, self.multi_item_embed = self.dual_gnn_update_embed()
            self.users = batch_data['users']
            self.pos_items=batch_data['pos_items']
            self.neg_items = batch_data['neg_items']
            self.u_g_emb = self.multi_user_embed[self.users,:,:];''''''
            self.pos_i_g_emb = self.multi_item_embed[self.pos_items,:,:]
            self.neg_i_g_emb = self.multi_item_embed[self.neg_items,:,:]

            # co attention
            self.pos_u_att_emb, self.pos_i_att_emb, self.neg_u_att_emb, self.neg_i_att_emb = self.interaction()

            # bpr_loss

            return [self.pos_u_att_emb, self.pos_i_att_emb, self.neg_u_att_emb, self.neg_i_att_emb,self.l2_reg]
        else:
            with torch.no_grad():
                if self.multi_item_embed == None:
                    self.multi_user_embed, self.multi_item_embed = self.dual_gnn_update_embed()
                self.users = batch_data['users']
                self.pos_items=batch_data['pos_items']
                self.u_g_emb = self.multi_user_embed[self.users,:,:]
                self.pos_i_g_emb = self.multi_item_embed[self.pos_items,:,:]
                self.pred_op = self.batch_pred(self.u_g_emb, self.pos_i_g_emb)
                return self.pred_op

    def interaction(self):
        if self.interaction_type == "co_attention":
            return self.co_attention()
        if self.interaction_type == 'han':
            return self.han()


    def bpr(self, pu, pi, nu, ni):
        # print("******************loss*********************")
        pos_pred = torch.multiply(pu, pi).sum(dim=1)
        neg_pred = torch.multiply(nu, ni).sum(dim=1)
        _mf_loss = F.softplus(-(pos_pred - neg_pred)).sum(dim=0)

        _emb_loss = torch.multiply(pu,pu).sum(dim=(0,1)) + torch.multiply(pi,pi).sum(dim=(0,1)) +\
                    torch.multiply(nu,nu).sum(dim=(0,1)) + torch.multiply(ni,ni).sum(dim=(0,1))
        _emb_loss = _emb_loss/2
        return _mf_loss, _emb_loss


    def batch_pred(self, u, i):
        global batch_rating
        if self.interaction_type == "co_attention":
            # uu = self.layer_trans_u(u)
            # ii = self.layer_trans_i(i)
            uu = u
            ii = i


            att_embed_1 ,att_embed_2 = self.co_attention_pair(uu,ii)

            batch_rating = torch.bmm(torch.unsqueeze(att_embed_1,dim=1), torch.unsqueeze(att_embed_2,dim=2))
            batch_rating = torch.squeeze(batch_rating)
        if self.interaction_type == 'han':
            #  感觉这样新建了var,和之前训练的是两套??
            uu, _ = self.SimpleAttLayer_user(u,
                                             return_alphas=True)
            ii, _ = self.SimpleAttLayer_item(i,
                                             return_alphas=True)
            batch_rating = torch.bmm(torch.unsqueeze(uu,dim=1), torch.unsqueeze(ii,dim=2))
            batch_rating = torch.squeeze(batch_rating)
        return batch_rating
    ############################################################co-attention

    def co_attention(self):
        #         regularizer = tf.contrib.layers.l2_regularizer(1e-5)
        # regularizer = torch.nn..layers.l2_regularizer(0.99)

        proj_u_g_emb = self.u_g_emb
        att_embed_1, att_embed_2 = self.co_attention_pair(self.u_g_emb,self.pos_i_g_emb)
        att_embed_3 , att_embed_4 = self.co_attention_pair(self.u_g_emb,self.neg_i_g_emb)

        return att_embed_1, att_embed_2, att_embed_3, att_embed_4

    def co_attention_pair(self,u,i):

        global U_mp_att_pos, I_mp_att_pos
        proj_u_g_emb = self.layer_trans_u(u)
        proj_pos_i_g_emb = self.layer_trans_i(i)

        # proj_u_g_emb = (u)
        # proj_pos_i_g_emb = (i)
        M_tmp = torch.reshape(torch.matmul(torch.reshape(proj_u_g_emb, [-1, cfg.emb_dim]), self.A), [-1, self.num_mp_1, cfg.emb_dim])
        M_pos = torch.bmm(M_tmp, proj_pos_i_g_emb.permute(0,2,1))
        if self.co_attention_type == 'max':
            U_mp_att_pos = F.softmax(input=torch.max(M_pos,dim=2).values,dim=1)  # [bs, num_mp_1]
            I_mp_att_pos = F.softmax(input=torch.max(M_pos,dim=1).values,dim=1)  # [bs, num_mp_2]

        if self.co_attention_type == 'mean':
            U_mp_att_pos = F.softmax(input= torch.mean(M_pos,dim=2),dim=1)  # [bs, num_mp_1]
            I_mp_att_pos = F.softmax(input=torch.mean(M_pos,dim=1),dim=1)  # [bs, num_mp_2]


        att_embed_1 = torch.squeeze(torch.matmul(proj_u_g_emb.transpose(1,2),  # [bs, dim, num_mp_1]
                                           torch.unsqueeze(U_mp_att_pos, -1)))  # [bs, num_mp_1, 1]
        att_embed_2 = torch.squeeze(torch.matmul(proj_pos_i_g_emb.transpose(1,2),  # [bs, dim, num_mp_1]
                                           torch.unsqueeze(I_mp_att_pos, -1)))

        return att_embed_1,att_embed_2



    def dual_gnn_update_embed(self):

        mp_user_embed_list = self.gnn_update_embed(
            self.init_user_embed, self.U_adj_list, self.u_gnn_depth[1:])

        mp_item_embed_list = self.gnn_update_embed(
            self.init_item_embed, self.I_adj_list, self.i_gnn_depth[1:])

        if self.U_MP[0] == self.I_MP[0][::-1]:
            # print(self.I_MP[0][::-1])
            U_embed_via_UI, I_embed_via_UI = self.UI_gnn_update(self.init_user_embed,
                                                                self.init_item_embed,
                                                                self.UIUI,
                                                                self.u_gnn_depth[0])
            mp_user_embed_list.append(U_embed_via_UI)

            mp_item_embed_list.append(I_embed_via_UI)

        multi_user_embed = torch.cat(mp_user_embed_list, dim=1).float()

        multi_item_embed = torch.cat(mp_item_embed_list, dim=1).float()

        return multi_user_embed, multi_item_embed

    def UI_gnn_update(self, user_embed, item_embed, UIUI, depth):
        global ui_gnn_embed
        ui_embed = torch.cat([user_embed, item_embed], dim=0)
        if self.gnn_type == 'gat':
            ui_gnn_embed = self.gat(UIUI, ui_embed, depth)
        if self.gnn_type == 'simple':
            ui_gnn_embed = self.simple(UIUI, ui_embed, depth)
        if self.gnn_type == 'ngcf':
            ui_gnn_embed = self.ngcf(UIUI, ui_embed, depth)
        ui_gnn_embed = torch.unsqueeze(ui_gnn_embed,dim=1)
        u_gnn_via_ui, i_gnn_via_ui = torch.split(ui_gnn_embed, [self.n_users, self.n_items], dim=0)

        return u_gnn_via_ui, i_gnn_via_ui

    def gnn_update_embed(self, fea, adj_list, depth_list):
        embed_list = []
        for idx, adj in enumerate(adj_list):
            if self.gnn_type == 'ngcf':
                tmp = self.ngcf(adj, fea, depth_list[idx])
            embed_list.append(torch.unsqueeze(tmp,dim=1))
        return embed_list

    def ngcf(self, adj, fea, depth):
        h = fea
        embed_list = [h]
        for i in range(depth):
            h = self.ngcf_layer(adj, h, self.coef_drop[i])
            embed_list.append(h)
        embed = torch.cat(embed_list,dim=1)
        # res = torch.mean(embed_list,dim=1)
        return embed

    # ngcf_layer
    def ngcf_layer(self, adj, fea, drop):
        ego_embeddings = fea
        # side_embeddings = torch.matmul(torch.from_numpy(adj).to(device).float(), fea.to(device).float())

        arr = []
        res = []
        for i in range(adj.shape[0]//1000 +1):
            arr.append(adj[1000*i:1000*(i+1),:])
            res.append(torch.matmul(torch.from_numpy(arr[i]).to(device).to(float),fea.to(device).to(float)))
        side_embeddings = torch.cat(res,dim=0)
        arr.clear()
        res.clear()

        sum_embeddings = self.ngcf_Linear1(fea.float())
        bi_embeddings = torch.multiply(ego_embeddings,
                                    side_embeddings)
        bi_embeddings = self.ngcf_Linear2(bi_embeddings.float())
        ego_embeddings = sum_embeddings + bi_embeddings

        ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(ego_embeddings)
        ego_embeddings  = self.drop(ego_embeddings)
        norm_embeddings = F.normalize(ego_embeddings,p=2, dim=1)
        return norm_embeddings  # ego_embeddings

    def han(self):
        # u [bs, num_mp, emb_dim]
        # print(self.u_g_emb.shape)
        final_u_embed, self.u_att_val = self.SimpleAttLayer_user(self.u_g_emb, return_alphas=True)
        final_pos_i_embed, self.pos_i_att_val = self.SimpleAttLayer_item(self.pos_i_g_emb,
                                                                         return_alphas=True)
        final_neg_i_embed, self.neg_i_att_val = self.SimpleAttLayer_item(self.neg_i_g_emb,
                                                                         return_alphas=True)
        return final_u_embed, final_pos_i_embed, final_u_embed, final_neg_i_embed

    def SimpleAttLayer_item(self, inputs, return_alphas=False):

        v = torch.tanh(self.item_Linear(inputs))
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = torch.tensordot(v, self.item_i_omega, dims=([2],[0]))  # (B,T) shape
        alphas = F.softmax(vu, 1)  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = torch.matmul(torch.transpose(inputs,1,2) , torch.unsqueeze(alphas, -1))
        output = torch.squeeze(output)
        if not return_alphas:
            return output
        else:
            return output, alphas

    def SimpleAttLayer_user(self, inputs, return_alphas=False):

        v = torch.tanh(self.user_Linear(inputs))

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = torch.tensordot(v, self.user_u_omega,dims=([2],[0])) # (B,T) shape

        alphas = F.softmax(vu, 1)  # (B,T) shape


        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = torch.matmul(torch.transpose(inputs,1,2) ,torch.unsqueeze(alphas, -1))
        output = torch.squeeze(output)
        if not return_alphas:
            return output
        else:
            return output, alphas

    def get_han_att(self, sess, feed_dict):

        return sess.run([self.u_att_val, self.pos_i_att_val],
                        feed_dict=feed_dict)

if __name__ == "__main__":
    # small test
    from config import config_ml_100k as cfg

    model = HGRec(cfg,
                     U_adj_list=[np.ones([10, 10], dtype=np.float32)],
                     I_adj_list=[np.ones([20, 20], dtype=np.float32)],
                     UIUI=np.ones([10 + 20, 10 + 20])
                     )
zzzzz