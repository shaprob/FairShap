" Importing packages "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb
from scipy.special import bernoulli
from itertools import chain, combinations
import numpy.matlib

# Some parameters
#Adult dataset: figsize=(5, 7), axs[ii-1].text(-0.04, 0.10 - 0.27 - 0.05, attr_names[ii-1])

def nParam_kAdd(kAdd,nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb

    
def powerset(iterable,nAttr):
    '''Return the powerset of a set of m attributes
    powerset([1,2,..., m],m) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m) ... (1, ..., m)
    powerset([1,2,..., m],2) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(nAttr+1))


def tr_shap2game(nAttr):
    '''Return the transformation matrix from Shapley interaction indices, given a k_additive model, to game'''
    nBern = bernoulli(nAttr) #Números de Bernoulli
    k_add_numb = nParam_kAdd(nAttr,nAttr)
    
    coalit = np.zeros((k_add_numb,nAttr))
    
    for i,s in enumerate(powerset(range(nAttr),nAttr)):
        s = list(s)
        coalit[i,s] = 1
        
    matrix_shap2game = np.zeros((k_add_numb,k_add_numb))
    for i in range(coalit.shape[0]):
        for i2 in range(k_add_numb):
            aux2 = int(sum(coalit[i2,:]))
            aux3 = int(sum(coalit[i,:] * coalit[i2,:]))
            aux4 = 0
            for i3 in range(int(aux3+1)):
                aux4 += comb(aux3, i3) * nBern[aux2-i3]
            matrix_shap2game[i,i2] = aux4
    return matrix_shap2game

def contr_rates(rate,rate_sens,rate_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,name_rate):
    " This function calculates the Shapley values for True/False positive/negative rates "
    " It also plots the results "
    
    shapley_all, shapley_sens, shapley_priv = np.zeros((rate.shape)), np.zeros((rate_sens.shape)), np.zeros((rate_sens.shape))
    
    ''' Comparing the overall, protected (sensible) and privileged groups along thresholds '''
    for kk in range(rate.shape[2]):
        shapley_all[:,:,kk] = transf_matrix @ rate[:,:,kk]
        shapley_sens[:,:,kk] = transf_matrix @ rate_sens[:,:,kk]
        shapley_priv[:,:,kk] = transf_matrix @ rate_priv[:,:,kk]
    
    shapley_all, shapley_all_std = np.mean(shapley_all[1:nAttr+1,:,:],axis=2), np.std(shapley_all[1:nAttr+1,:,:],axis=2)  
    shapley_sens, shapley_sens_std = np.mean(shapley_sens[1:nAttr+1,:,:],axis=2), np.std(shapley_sens[1:nAttr+1,:,:],axis=2)
    shapley_priv, shapley_priv_std = np.mean(shapley_priv[1:nAttr+1,:,:],axis=2), np.std(shapley_priv[1:nAttr+1,:,:],axis=2)  
    
    min_shap = 1.1*np.min(np.vstack([shapley_all-shapley_all_std,shapley_sens-shapley_sens_std,shapley_priv-shapley_priv_std]))
    max_shap = 1.1*np.max(np.vstack([shapley_all+shapley_all_std,shapley_sens+shapley_sens_std,shapley_priv+shapley_priv_std]))
             
    ''' Plot rates for overall, protected (sensible) and privileged groups along thresholds '''
    plt.show()
    plt.plot(thresh, np.mean(rate[-1,:,:],axis=1), 'k', thresh, np.mean(rate_sens[-1,:,:],axis=1), 'r', thresh, np.mean(rate_priv[-1,:,:],axis=1), 'b', thresh, np.flip(thresh), 'k--')
    plt.fill_between(thresh, np.mean(rate[-1,:,:],axis=1)-np.std(rate[-1,:,:],axis=1), np.mean(rate[-1,:,:],axis=1)+np.std(rate[-1,:,:],axis=1),alpha=0.3, color='k')
    plt.fill_between(thresh, np.mean(rate_sens[-1,:,:],axis=1)-np.std(rate_sens[-1,:,:],axis=1), np.mean(rate_sens[-1,:,:],axis=1)+np.std(rate_sens[-1,:,:],axis=1),alpha=0.3, color='r')
    plt.fill_between(thresh, np.mean(rate_priv[-1,:,:],axis=1)-np.std(rate_priv[-1,:,:],axis=1), np.mean(rate_priv[-1,:,:],axis=1)+np.std(rate_priv[-1,:,:],axis=1),alpha=0.3, color='b')
    plt.legend(['Overall', 'Protected group', 'Privileged group', 'Random classifier'])
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel(name_rate,fontsize=12)
    
    ''' Feature's contribution for overall, protected (sensible) and privileged groups along thresholds '''
    # Overall
    plt.show()
    fig = plt.figure(figsize=(5, 6))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(0,nAttr):
        axs[ii].plot(thresh,shapley_all[ii,:], label = name_rate, color='k')
        axs[ii].fill_between(thresh, shapley_all[ii,:]-shapley_all_std[ii,:], shapley_all[ii,:]+shapley_all_std[ii,:],alpha=0.3, color='k')
        axs[ii].set_ylim([min_shap, max_shap])
        axs[ii].text(0.05, 0.15, attr_names[ii])
        axs[ii].set_xlim([thresh[0], thresh[-1]])
        plt.xlabel('Threshold',fontsize=11)
        #if ii == 1:
        #    axs[ii-1].legend(loc="upper right",fontsize=12)
    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on %s' %name_rate,fontsize=11, va='center', rotation='vertical')
    
    # Protected
    plt.show()
    fig = plt.figure(figsize=(5, 6))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(0,nAttr):
        axs[ii].plot(thresh,shapley_sens[ii,:], label = name_rate, color='r')
        axs[ii].fill_between(thresh, shapley_sens[ii,:]-shapley_sens_std[ii,:], shapley_sens[ii,:]+shapley_sens_std[ii,:],alpha=0.3, color='r')
        axs[ii].set_ylim([min_shap, max_shap])
        axs[ii].text(0.05, 0.15, attr_names[ii])
        axs[ii].set_xlim([thresh[0], thresh[-1]])
        plt.xlabel('Threshold',fontsize=11)
        #if ii == 1:
        #    axs[ii-1].legend(loc="upper right",fontsize=12)
    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on %s' %name_rate,fontsize=11, va='center', rotation='vertical')
    
    # Privileged
    plt.show()
    fig = plt.figure(figsize=(5, 6))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(0,nAttr):
        axs[ii].plot(thresh,shapley_priv[ii,:], label = name_rate, color='b')
        axs[ii].fill_between(thresh, shapley_priv[ii,:]-shapley_priv_std[ii,:], shapley_priv[ii,:]+shapley_priv_std[ii,:],alpha=0.3, color='b')
        axs[ii].set_ylim([min_shap, max_shap])
        axs[ii].text(0.05, 0.15, attr_names[ii])
        axs[ii].set_xlim([thresh[0], thresh[-1]])
        plt.xlabel('Threshold',fontsize=11)
        #if ii == 1:
        #    axs[ii-1].legend(loc="upper right",fontsize=12)
    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on %s' %name_rate,fontsize=11, va='center', rotation='vertical')
    
    # All together
    plt.show()
    fig = plt.figure(figsize=(5, 6))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(0,nAttr):
        axs[ii].plot(thresh,shapley_all[ii,:], color='k')
        axs[ii].plot(thresh,shapley_sens[ii,:], color='r')
        axs[ii].plot(thresh,shapley_priv[ii,:], label = name_rate, color='b')
        axs[ii].fill_between(thresh, shapley_all[ii,:]-shapley_all_std[ii,:], shapley_all[ii,:]+shapley_all_std[ii,:],alpha=0.3, color='k')
        axs[ii].fill_between(thresh, shapley_sens[ii,:]-shapley_sens_std[ii,:], shapley_sens[ii,:]+shapley_sens_std[ii,:],alpha=0.3, color='r')
        axs[ii].fill_between(thresh, shapley_priv[ii,:]-shapley_priv_std[ii,:], shapley_priv[ii,:]+shapley_priv_std[ii,:],alpha=0.3, color='b')
        axs[ii].set_ylim([min_shap, max_shap])
        axs[ii].text(0.05, 0.15, attr_names[ii])
        axs[ii].set_xlim([thresh[0], thresh[-1]])
        plt.xlabel('Threshold',fontsize=11)
        #if ii == 1:
        #    axs[ii-1].legend(loc="upper right",fontsize=12)
    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on %s' %name_rate,fontsize=11, va='center', rotation='vertical')
    fig.legend(['Overall', 'Protected group', 'Privileged group'], bbox_to_anchor =(0.45, 0.96), loc='upper center', ncol=3)
    
    return shapley_all, shapley_sens, shapley_priv


def statistical_parity(tp,fp,tp_sens,fp_sens,tp_priv,fp_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold):
    " This function calculates the Shapley values for Statistical Parity "
    " It also plots the results "
    
    sp = (tp + fp)/np.tile(nSens+nPriv,[2**nAttr,len(thresh),1])
    sp_sens = (tp_sens + fp_sens)/np.tile(nSens, [2**nAttr, len(thresh),1])
    sp_priv = (tp_priv + fp_priv)/np.tile(nPriv, [2**nAttr, len(thresh),1])
    StPa = np.abs(sp_sens - sp_priv)

    " Calculating Shapley values "
    shapley_StPa_all, shapley_StPa_sens, shapley_StPa_priv, shapley_StPa = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))
    for kk in range(nFold):
        shapley_StPa_all[:,:,kk] = transf_matrix @ sp[:,:,kk]
        shapley_StPa_sens[:,:,kk] = transf_matrix @ sp_sens[:,:,kk]
        shapley_StPa_priv[:,:,kk] = transf_matrix @ sp_priv[:,:,kk]
        shapley_StPa[:,:,kk] = transf_matrix @ StPa[:,:,kk]
        
    shapley_StPa_all_mean = np.mean(shapley_StPa_all,axis=2)
    shapley_StPa_sens_mean = np.mean(shapley_StPa_sens,axis=2)
    shapley_StPa_priv_mean = np.mean(shapley_StPa_priv,axis=2)
    shapley_StPa_mean = np.mean(shapley_StPa,axis=2)

    shapley_StPa_all_std = np.std(shapley_StPa_all,axis=2)
    shapley_StPa_sens_std = np.std(shapley_StPa_sens,axis=2)
    shapley_StPa_priv_std = np.std(shapley_StPa_priv,axis=2)
    shapley_StPa_std = np.std(shapley_StPa,axis=2)
    
    min_shap = 1.1*np.min(np.vstack([shapley_StPa_all_mean[1:nAttr+1,:],shapley_StPa_sens_mean[1:nAttr+1,:],shapley_StPa_priv_mean[1:nAttr+1,:]])-np.vstack([shapley_StPa_all_std[1:nAttr+1,:],shapley_StPa_sens_std[1:nAttr+1,:],shapley_StPa_priv_std[1:nAttr+1,:]]))
    max_shap = 1.1*np.max(np.vstack([shapley_StPa_all_mean[1:nAttr+1,:],shapley_StPa_sens_mean[1:nAttr+1,:],shapley_StPa_priv_mean[1:nAttr+1,:]])+np.vstack([shapley_StPa_all_std[1:nAttr+1,:],shapley_StPa_sens_std[1:nAttr+1,:],shapley_StPa_priv_std[1:nAttr+1,:]]))
    
    plt.show()
    fig = plt.figure(figsize=(5, 7))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_StPa_all_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_StPa_all_mean[ii,:] - shapley_StPa_all_std[ii,:], shapley_StPa_all_mean[ii,:] + shapley_StPa_all_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].plot(thresh,shapley_StPa_sens_mean[ii,:], label = 'Protected group', color='r')
        axs[ii-1].fill_between(thresh, shapley_StPa_sens_mean[ii,:] - shapley_StPa_sens_std[ii,:], shapley_StPa_sens_mean[ii,:] + shapley_StPa_sens_std[ii,:],alpha=0.3, color='r')
        axs[ii-1].plot(thresh,shapley_StPa_priv_mean[ii,:], label = 'Privileged group', color='b')
        axs[ii-1].fill_between(thresh, shapley_StPa_priv_mean[ii,:] - shapley_StPa_priv_std[ii,:], shapley_StPa_priv_mean[ii,:] + shapley_StPa_priv_std[ii,:],alpha=0.3, color='b')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(-0.04, 0.10, attr_names[ii-1])
        plt.xlabel('Threshold',fontsize=11)

    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on the probability of favorable outcome',fontsize=11, va='center', rotation='vertical')
    fig.legend(['Overall', 'Protected group', 'Privileged group'], bbox_to_anchor =(0.45, 0.96), loc='upper center', ncol=3)
   
    plt.show()
    plt.plot(thresh, np.mean(sp[-1,:,:],axis=1), 'k', thresh, np.mean(sp_sens[-1,:,:],axis=1), 'r', thresh, np.mean(sp_priv[-1,:,:],axis=1), 'b', thresh, np.flip(thresh), 'k--')
    plt.fill_between(thresh, np.mean(sp[-1,:,:],axis=1)-np.std(sp[-1,:,:],axis=1), np.mean(sp[-1,:,:],axis=1)+np.std(sp[-1,:,:],axis=1),alpha=0.3, color='k')
    plt.fill_between(thresh, np.mean(sp_sens[-1,:,:],axis=1)-np.std(sp_sens[-1,:,:],axis=1), np.mean(sp_sens[-1,:,:],axis=1)+np.std(sp_sens[-1,:,:],axis=1),alpha=0.3, color='r')
    plt.fill_between(thresh, np.mean(sp_priv[-1,:,:],axis=1)-np.std(sp_priv[-1,:,:],axis=1), np.mean(sp_priv[-1,:,:],axis=1)+np.std(sp_priv[-1,:,:],axis=1),alpha=0.3, color='b')
    plt.legend(['Overall', 'Protected group', 'Privileged group', 'Random classifier'])
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('Probability of favorable outcome',fontsize=12)

    plt.show()
    for ii in range(nAttr):
        plt.plot(thresh, shapley_StPa_mean[ii+1,:])
    plt.legend(attr_names,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Threshold',fontsize=11)
    plt.ylabel('Feature contribution on Statistical Parity',fontsize=11)
    
    min_shap = 1.1*np.min(shapley_StPa_mean[1:nAttr+1,:]-shapley_StPa_std[1:nAttr+1,:])
    max_shap = 1.1*np.max(shapley_StPa_mean[1:nAttr+1,:]+shapley_StPa_std[1:nAttr+1,:])
    
    plt.show()
    fig = plt.figure(figsize=(5, 4.5))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_StPa_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_StPa_mean[ii,:] - shapley_StPa_std[ii,:], shapley_StPa_mean[ii,:] + shapley_StPa_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(-0.04, 0.35, attr_names[ii-1])
        plt.xlabel('Threshold',fontsize=11)

    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on Statistical Parity',fontsize=11, va='center', rotation='vertical')
    
    plt.show()
    plt.plot(thresh, np.mean(StPa[-1,:,:],axis=1), 'k')
    plt.fill_between(thresh, np.mean(StPa[-1,:,:],axis=1)-np.std(StPa[-1,:,:],axis=1), np.mean(StPa[-1,:,:],axis=1)+np.std(StPa[-1,:,:],axis=1),alpha=0.3, color='k')
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('Statistical Parity',fontsize=12)
    
    return shapley_StPa_all_mean, shapley_StPa_sens_mean, shapley_StPa_priv_mean, shapley_StPa_mean, shapley_StPa_all_std, shapley_StPa_sens_std, shapley_StPa_priv_std, shapley_StPa_std

def predictive_parity(ppv,ppv_sens,ppv_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold):
    " This function calculates the Shapley values for Predictive Parity "
    " It also plots the results "
    
    PrPa = np.abs(ppv_sens - ppv_priv)

    " Calculating Shapley values "
    shapley_PrPa_all, shapley_PrPa_sens, shapley_PrPa_priv, shapley_PrPa = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))

    for kk in range(nFold):
        shapley_PrPa_all[:,:,kk] = transf_matrix @ ppv[:,:,kk]
        shapley_PrPa_sens[:,:,kk] = transf_matrix @ ppv_sens[:,:,kk]
        shapley_PrPa_priv[:,:,kk] = transf_matrix @ ppv_priv[:,:,kk]
        shapley_PrPa[:,:,kk] = transf_matrix @ PrPa[:,:,kk]
        
    shapley_PrPa_all_mean = np.mean(shapley_PrPa_all,axis=2)
    shapley_PrPa_sens_mean = np.mean(shapley_PrPa_sens,axis=2)
    shapley_PrPa_priv_mean = np.mean(shapley_PrPa_priv,axis=2)
    shapley_PrPa_mean = np.mean(shapley_PrPa,axis=2)

    shapley_PrPa_all_std = np.std(shapley_PrPa_all,axis=2)
    shapley_PrPa_sens_std = np.std(shapley_PrPa_sens,axis=2)
    shapley_PrPa_priv_std = np.std(shapley_PrPa_priv,axis=2)
    shapley_PrPa_std = np.std(shapley_PrPa,axis=2)
    
    min_shap = 1.1*np.min(np.vstack([shapley_PrPa_all_mean[1:nAttr+1,:],shapley_PrPa_sens_mean[1:nAttr+1,:],shapley_PrPa_priv_mean[1:nAttr+1,:]])-np.vstack([shapley_PrPa_all_std[1:nAttr+1,:],shapley_PrPa_sens_std[1:nAttr+1,:],shapley_PrPa_priv_std[1:nAttr+1,:]]))
    max_shap = 1.1*np.max(np.vstack([shapley_PrPa_all_mean[1:nAttr+1,:],shapley_PrPa_sens_mean[1:nAttr+1,:],shapley_PrPa_priv_mean[1:nAttr+1,:]])+np.vstack([shapley_PrPa_all_std[1:nAttr+1,:],shapley_PrPa_sens_std[1:nAttr+1,:],shapley_PrPa_priv_std[1:nAttr+1,:]]))
    
    plt.show()
    fig = plt.figure(figsize=(5, 6))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_PrPa_all_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_PrPa_all_mean[ii,:] - shapley_PrPa_all_std[ii,:], shapley_PrPa_all_mean[ii,:] + shapley_PrPa_all_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].plot(thresh,shapley_PrPa_sens_mean[ii,:], label = 'Protected group', color='r')
        axs[ii-1].fill_between(thresh, shapley_PrPa_sens_mean[ii,:] - shapley_PrPa_sens_std[ii,:], shapley_PrPa_sens_mean[ii,:] + shapley_PrPa_sens_std[ii,:],alpha=0.3, color='r')
        axs[ii-1].plot(thresh,shapley_PrPa_priv_mean[ii,:], label = 'Privileged group', color='b')
        axs[ii-1].fill_between(thresh, shapley_PrPa_priv_mean[ii,:] - shapley_PrPa_priv_std[ii,:], shapley_PrPa_priv_mean[ii,:] + shapley_PrPa_priv_std[ii,:],alpha=0.3, color='b')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(0.05, 0.20, attr_names[ii-1])
        #○axs[ii-1].set_title(attr_names[ii-1],horizontalalignment='right',verticalalignment='center')
        plt.xlabel('Threshold',fontsize=11)
        #if ii == 1:
        #    axs[ii-1].legend(loc="upper right",fontsize=12)
    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on the PPV',fontsize=11, va='center', rotation='vertical')
    fig.legend(['Overall', 'Protected group', 'Privileged group'], bbox_to_anchor =(0.45, 0.96), loc='upper center', ncol=3)
  
    plt.show()
    plt.plot(thresh, np.mean(ppv[-1,:,:],axis=1), 'k', thresh, np.mean(ppv_sens[-1,:,:],axis=1), 'r', thresh, np.mean(ppv_priv[-1,:,:],axis=1), 'b', thresh, np.flip(thresh), 'k--')
    plt.legend(['Overall', 'Protected group', 'Privileged group', 'Random classifier'])
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('PPV',fontsize=12)

    plt.show()
    for ii in range(nAttr):
        plt.plot(thresh, shapley_PrPa_mean[ii+1,:])
    plt.legend(attr_names,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Threshold',fontsize=11)
    plt.ylabel('Predictive Parity',fontsize=11)
    
    return shapley_PrPa_all_mean, shapley_PrPa_sens_mean, shapley_PrPa_priv_mean, shapley_PrPa_mean, shapley_PrPa_all_std, shapley_PrPa_sens_std, shapley_PrPa_priv_std, shapley_PrPa_std

def predictive_equality(fpr,fpr_sens,fpr_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold):
    " This function calculates the Shapley values for Predictive Equality "
    " It also plots the results "
    
    PrEq = np.abs(fpr_sens - fpr_priv)

    " Calculating Shapley values "
    shapley_PrEq_all, shapley_PrEq_sens, shapley_PrEq_priv, shapley_PrEq = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))

    for kk in range(nFold):
        shapley_PrEq_all[:,:,kk] = transf_matrix @ fpr[:,:,kk]
        shapley_PrEq_sens[:,:,kk] = transf_matrix @ fpr_sens[:,:,kk]
        shapley_PrEq_priv[:,:,kk] = transf_matrix @ fpr_priv[:,:,kk]
        shapley_PrEq[:,:,kk] = transf_matrix @ PrEq[:,:,kk]
        
    shapley_PrEq_all_mean = np.mean(shapley_PrEq_all,axis=2)
    shapley_PrEq_sens_mean = np.mean(shapley_PrEq_sens,axis=2)
    shapley_PrEq_priv_mean = np.mean(shapley_PrEq_priv,axis=2)
    shapley_PrEq_mean = np.mean(shapley_PrEq,axis=2)

    shapley_PrEq_all_std = np.std(shapley_PrEq_all,axis=2)
    shapley_PrEq_sens_std = np.std(shapley_PrEq_sens,axis=2)
    shapley_PrEq_priv_std = np.std(shapley_PrEq_priv,axis=2)
    shapley_PrEq_std = np.std(shapley_PrEq,axis=2)
    
    min_shap = 1.1*np.min(np.vstack([shapley_PrEq_all_mean[1:nAttr+1,:],shapley_PrEq_sens_mean[1:nAttr+1,:],shapley_PrEq_priv_mean[1:nAttr+1,:]])-np.vstack([shapley_PrEq_all_std[1:nAttr+1,:],shapley_PrEq_sens_std[1:nAttr+1,:],shapley_PrEq_priv_std[1:nAttr+1,:]]))
    max_shap = 1.1*np.max(np.vstack([shapley_PrEq_all_mean[1:nAttr+1,:],shapley_PrEq_sens_mean[1:nAttr+1,:],shapley_PrEq_priv_mean[1:nAttr+1,:]])+np.vstack([shapley_PrEq_all_std[1:nAttr+1,:],shapley_PrEq_sens_std[1:nAttr+1,:],shapley_PrEq_priv_std[1:nAttr+1,:]]))
    
    plt.show()
    fig = plt.figure(figsize=(5, 7))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_PrEq_all_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_PrEq_all_mean[ii,:] - shapley_PrEq_all_std[ii,:], shapley_PrEq_all_mean[ii,:] + shapley_PrEq_all_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].plot(thresh,shapley_PrEq_sens_mean[ii,:], label = 'Protected group', color='r')
        axs[ii-1].fill_between(thresh, shapley_PrEq_sens_mean[ii,:] - shapley_PrEq_sens_std[ii,:], shapley_PrEq_sens_mean[ii,:] + shapley_PrEq_sens_std[ii,:],alpha=0.3, color='r')
        axs[ii-1].plot(thresh,shapley_PrEq_priv_mean[ii,:], label = 'Privileged group', color='b')
        axs[ii-1].fill_between(thresh, shapley_PrEq_priv_mean[ii,:] - shapley_PrEq_priv_std[ii,:], shapley_PrEq_priv_mean[ii,:] + shapley_PrEq_priv_std[ii,:],alpha=0.3, color='b')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(-0.04, 0.05, attr_names[ii-1])
        #○axs[ii-1].set_title(attr_names[ii-1],horizontalalignment='right',verticalalignment='center')
        plt.xlabel('Threshold',fontsize=11)
        #if ii == 1:
        #    axs[ii-1].legend(loc="upper right",fontsize=12)
    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on FPR',fontsize=11, va='center', rotation='vertical')
    fig.legend(['Overall', 'Protected group', 'Privileged group'], bbox_to_anchor =(0.45, 0.96), loc='upper center', ncol=3)
   
    plt.show()
    plt.plot(thresh, np.mean(fpr[-1,:,:],axis=1), 'k', thresh, np.mean(fpr_sens[-1,:,:],axis=1), 'r', thresh, np.mean(fpr_priv[-1,:,:],axis=1), 'b', thresh, np.flip(thresh), 'k--')
    plt.fill_between(thresh, np.mean(fpr[-1,:,:],axis=1)-np.std(fpr[-1,:,:],axis=1), np.mean(fpr[-1,:,:],axis=1)+np.std(fpr[-1,:,:],axis=1),alpha=0.3, color='k')
    plt.fill_between(thresh, np.mean(fpr_sens[-1,:,:],axis=1)-np.std(fpr_sens[-1,:,:],axis=1), np.mean(fpr_sens[-1,:,:],axis=1)+np.std(fpr_sens[-1,:,:],axis=1),alpha=0.3, color='r')
    plt.fill_between(thresh, np.mean(fpr_priv[-1,:,:],axis=1)-np.std(fpr_priv[-1,:,:],axis=1), np.mean(fpr_priv[-1,:,:],axis=1)+np.std(fpr_priv[-1,:,:],axis=1),alpha=0.3, color='b')
    plt.legend(['Overall', 'Protected group', 'Privileged group', 'Random classifier'])
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('FPR',fontsize=12)
    
    plt.show()
    for ii in range(nAttr):
        plt.plot(thresh, shapley_PrEq_mean[ii+1,:])
    plt.legend(attr_names,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Threshold',fontsize=11)
    plt.ylabel('Feature contribution on Predictive Equality',fontsize=11)
    
    min_shap = 1.1*np.min(shapley_PrEq_mean[1:nAttr+1,:]-shapley_PrEq_std[1:nAttr+1,:])
    max_shap = 1.1*np.max(shapley_PrEq_mean[1:nAttr+1,:]+shapley_PrEq_std[1:nAttr+1,:])
    
    plt.show()
    fig = plt.figure(figsize=(5, 7))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_PrEq_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_PrEq_mean[ii,:] - shapley_PrEq_std[ii,:], shapley_PrEq_mean[ii,:] + shapley_PrEq_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(-0.04, 0.27, attr_names[ii-1])
        plt.xlabel('Threshold',fontsize=11)

    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on Predictive Equality',fontsize=11, va='center', rotation='vertical')
    
    plt.show()
    plt.plot(thresh, np.mean(PrEq[-1,:,:],axis=1), 'k')
    plt.fill_between(thresh, np.mean(PrEq[-1,:,:],axis=1)-np.std(PrEq[-1,:,:],axis=1), np.mean(PrEq[-1,:,:],axis=1)+np.std(PrEq[-1,:,:],axis=1),alpha=0.3, color='k')
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('Predictive Equality',fontsize=12)

    
    return shapley_PrEq_all_mean, shapley_PrEq_sens_mean, shapley_PrEq_priv_mean, shapley_PrEq_mean, shapley_PrEq_all_std, shapley_PrEq_sens_std, shapley_PrEq_priv_std, shapley_PrEq_std

def equal_opportunity(tpr,tpr_sens,tpr_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold):
    " This function calculates the Shapley values for Equal Opportunity "
    " It also plots the results "
    
    EqOp = np.abs(tpr_sens - tpr_priv)

    " Calculating Shapley values "
    shapley_EqOp_all, shapley_EqOp_sens, shapley_EqOp_priv, shapley_EqOp = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))

    for kk in range(nFold):
        shapley_EqOp_all[:,:,kk] = transf_matrix @ tpr[:,:,kk]
        shapley_EqOp_sens[:,:,kk] = transf_matrix @ tpr_sens[:,:,kk]
        shapley_EqOp_priv[:,:,kk] = transf_matrix @ tpr_priv[:,:,kk]
        shapley_EqOp[:,:,kk] = transf_matrix @ EqOp[:,:,kk]
        
    shapley_EqOp_all_mean = np.mean(shapley_EqOp_all,axis=2)
    shapley_EqOp_sens_mean = np.mean(shapley_EqOp_sens,axis=2)
    shapley_EqOp_priv_mean = np.mean(shapley_EqOp_priv,axis=2)
    shapley_EqOp_mean = np.mean(shapley_EqOp,axis=2)

    shapley_EqOp_all_std = np.std(shapley_EqOp_all,axis=2)
    shapley_EqOp_sens_std = np.std(shapley_EqOp_sens,axis=2)
    shapley_EqOp_priv_std = np.std(shapley_EqOp_priv,axis=2)
    shapley_EqOp_std = np.std(shapley_EqOp,axis=2)
    
    min_shap = 1.1*np.min(np.vstack([shapley_EqOp_all_mean[1:nAttr+1,:],shapley_EqOp_sens_mean[1:nAttr+1,:],shapley_EqOp_priv_mean[1:nAttr+1,:]])-np.vstack([shapley_EqOp_all_std[1:nAttr+1,:],shapley_EqOp_sens_std[1:nAttr+1,:],shapley_EqOp_priv_std[1:nAttr+1,:]]))
    max_shap = 1.1*np.max(np.vstack([shapley_EqOp_all_mean[1:nAttr+1,:],shapley_EqOp_sens_mean[1:nAttr+1,:],shapley_EqOp_priv_mean[1:nAttr+1,:]])+np.vstack([shapley_EqOp_all_std[1:nAttr+1,:],shapley_EqOp_sens_std[1:nAttr+1,:],shapley_EqOp_priv_std[1:nAttr+1,:]]))
    
    plt.show()
    fig = plt.figure(figsize=(5, 7))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_EqOp_all_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_EqOp_all_mean[ii,:] - shapley_EqOp_all_std[ii,:], shapley_EqOp_all_mean[ii,:] + shapley_EqOp_all_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].plot(thresh,shapley_EqOp_sens_mean[ii,:], label = 'Protected group', color='r')
        axs[ii-1].fill_between(thresh, shapley_EqOp_sens_mean[ii,:] - shapley_EqOp_sens_std[ii,:], shapley_EqOp_sens_mean[ii,:] + shapley_EqOp_sens_std[ii,:],alpha=0.3, color='r')
        axs[ii-1].plot(thresh,shapley_EqOp_priv_mean[ii,:], label = 'Privileged group', color='b')
        axs[ii-1].fill_between(thresh, shapley_EqOp_priv_mean[ii,:] - shapley_EqOp_priv_std[ii,:], shapley_EqOp_priv_mean[ii,:] + shapley_EqOp_priv_std[ii,:],alpha=0.3, color='b')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(-0.04, 0.20, attr_names[ii-1])
        #○axs[ii-1].set_title(attr_names[ii-1],horizontalalignment='right',verticalalignment='center')
        plt.xlabel('Threshold',fontsize=11)
        #if ii == 1:
        #    axs[ii-1].legend(loc="upper right",fontsize=12)
    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on TPR',fontsize=11, va='center', rotation='vertical')
    fig.legend(['Overall', 'Protected group', 'Privileged group'], bbox_to_anchor =(0.45, 0.96), loc='upper center', ncol=3)
   
    plt.show()
    plt.plot(thresh, np.mean(tpr[-1,:,:],axis=1), 'k', thresh, np.mean(tpr_sens[-1,:,:],axis=1), 'r', thresh, np.mean(tpr_priv[-1,:,:],axis=1), 'b', thresh, np.flip(thresh), 'k--')
    plt.fill_between(thresh, np.mean(tpr[-1,:,:],axis=1)-np.std(tpr[-1,:,:],axis=1), np.mean(tpr[-1,:,:],axis=1)+np.std(tpr[-1,:,:],axis=1),alpha=0.3, color='k')
    plt.fill_between(thresh, np.mean(tpr_sens[-1,:,:],axis=1)-np.std(tpr_sens[-1,:,:],axis=1), np.mean(tpr_sens[-1,:,:],axis=1)+np.std(tpr_sens[-1,:,:],axis=1),alpha=0.3, color='r')
    plt.fill_between(thresh, np.mean(tpr_priv[-1,:,:],axis=1)-np.std(tpr_priv[-1,:,:],axis=1), np.mean(tpr_priv[-1,:,:],axis=1)+np.std(tpr_priv[-1,:,:],axis=1),alpha=0.3, color='b')
    plt.legend(['Overall', 'Protected group', 'Privileged group', 'Random classifier'])
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('TPR',fontsize=12)
    
    plt.show()
    for ii in range(nAttr):
        plt.plot(thresh, shapley_EqOp_mean[ii+1,:])
    plt.legend(attr_names,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Threshold',fontsize=11)
    plt.ylabel('Feature contribution on Equal Opportunity',fontsize=11)
    
    min_shap = 1.1*np.min(shapley_EqOp_mean[1:nAttr+1,:]-shapley_EqOp_std[1:nAttr+1,:])
    max_shap = 1.1*np.max(shapley_EqOp_mean[1:nAttr+1,:]+shapley_EqOp_std[1:nAttr+1,:])
    
    plt.show()
    fig = plt.figure(figsize=(5, 7))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_EqOp_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_EqOp_mean[ii,:] - shapley_EqOp_std[ii,:], shapley_EqOp_mean[ii,:] + shapley_EqOp_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(-0.04, 0.27, attr_names[ii-1])
        plt.xlabel('Threshold',fontsize=11)

    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on Equal Opportunity',fontsize=11, va='center', rotation='vertical')
    
    
    plt.show()
    plt.plot(thresh, np.mean(EqOp[-1,:,:],axis=1), 'k')
    plt.fill_between(thresh, np.mean(EqOp[-1,:,:],axis=1)-np.std(EqOp[-1,:,:],axis=1), np.mean(EqOp[-1,:,:],axis=1)+np.std(EqOp[-1,:,:],axis=1),alpha=0.3, color='k')
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('Equal Opportunity',fontsize=12)

    return shapley_EqOp_all_mean, shapley_EqOp_sens_mean, shapley_EqOp_priv_mean, shapley_EqOp_mean, shapley_EqOp_all_std, shapley_EqOp_sens_std, shapley_EqOp_priv_std, shapley_EqOp_std

def equalized_odds(tpr,fpr,tpr_sens,fpr_sens,tpr_priv,fpr_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold):
    " This function calculates the Shapley values for Equalized Odds "
    " It also plots the results "
    
    EqOd = np.abs(tpr_sens - tpr_priv) + np.abs(fpr_sens - fpr_priv)

    " Calculating Shapley values "
    shapley_EqOd_all, shapley_EqOd_sens, shapley_EqOd_priv, shapley_EqOd = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))
    for kk in range(nFold):
        shapley_EqOd[:,:,kk] = transf_matrix @ EqOd[:,:,kk]
        
    shapley_EqOd_mean = np.mean(shapley_EqOd,axis=2)

    shapley_EqOd_std = np.std(shapley_EqOd,axis=2)
    
    min_shap = np.min(shapley_EqOd_mean[1:nAttr+1,:]-shapley_EqOd_std[1:nAttr+1,:])
    max_shap = np.max(shapley_EqOd_mean[1:nAttr+1,:]+shapley_EqOd_std[1:nAttr+1,:])
    
    plt.show()
    for ii in range(nAttr):
        plt.plot(thresh, shapley_EqOd_mean[ii+1,:])
    plt.legend(attr_names,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Threshold',fontsize=11)
    plt.ylabel('Feature contribution on Equalized Odds',fontsize=11)
    
    min_shap = 1.1*np.min(shapley_EqOd_mean[1:nAttr+1,:]-shapley_EqOd_std[1:nAttr+1,:])
    max_shap = 1.1*np.max(shapley_EqOd_mean[1:nAttr+1,:]+shapley_EqOd_std[1:nAttr+1,:])
    
    plt.show()
    fig = plt.figure(figsize=(5, 7))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_EqOd_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_EqOd_mean[ii,:] - shapley_EqOd_std[ii,:], shapley_EqOd_mean[ii,:] + shapley_EqOd_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(-0.04, 0.27, attr_names[ii-1])
        plt.xlabel('Threshold',fontsize=11)

    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on Equalized Odds',fontsize=11, va='center', rotation='vertical')
    
    plt.show()
    plt.plot(thresh, np.mean(EqOd[-1,:,:],axis=1), 'k')
    plt.fill_between(thresh, np.mean(EqOd[-1,:,:],axis=1)-np.std(EqOd[-1,:,:],axis=1), np.mean(EqOd[-1,:,:],axis=1)+np.std(EqOd[-1,:,:],axis=1),alpha=0.3, color='k')
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('Equalized Odds',fontsize=12)
    
    return shapley_EqOd_mean, shapley_EqOd_std

def conditional_accuracy_equality(ppv,ppv_sens,ppv_priv,npv,npv_sens,npv_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold):
    " This function calculates the Shapley values for Conditional Use Accuracy Equality "
    " It also plots the results "
    
    CAEq = np.abs(ppv_sens - ppv_priv) + np.abs(npv_sens - npv_priv)

    " Calculating Shapley values "
    shapley_CAEq_npv, shapley_CAEq_npv_sens, shapley_CAEq_npv_priv, shapley_CAEq = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))

    for kk in range(nFold):
        shapley_CAEq_npv[:,:,kk] = transf_matrix @ npv[:,:,kk]
        shapley_CAEq_npv_sens[:,:,kk] = transf_matrix @ npv_sens[:,:,kk]
        shapley_CAEq_npv_priv[:,:,kk] = transf_matrix @ npv_priv[:,:,kk]
        shapley_CAEq[:,:,kk] = transf_matrix @ CAEq[:,:,kk]
        
    shapley_CAEq_npv_mean = np.mean(shapley_CAEq_npv,axis=2)
    shapley_CAEq_npv_sens_mean = np.mean(shapley_CAEq_npv_sens,axis=2)
    shapley_CAEq_npv_priv_mean = np.mean(shapley_CAEq_npv_priv,axis=2)
    shapley_CAEq_mean = np.mean(shapley_CAEq,axis=2)

    shapley_CAEq_npv_std = np.std(shapley_CAEq_npv,axis=2)
    shapley_CAEq_npv_sens_std = np.std(shapley_CAEq_npv_sens,axis=2)
    shapley_CAEq_npv_priv_std = np.std(shapley_CAEq_npv_priv,axis=2)
    shapley_CAEq_std = np.std(shapley_CAEq,axis=2)
    
    min_shap = 1.1*np.min(np.vstack([shapley_CAEq_npv_mean[1:nAttr+1,:],shapley_CAEq_npv_sens_mean[1:nAttr+1,:],shapley_CAEq_npv_priv_mean[1:nAttr+1,:]])-np.vstack([shapley_CAEq_npv_std[1:nAttr+1,:],shapley_CAEq_npv_sens_std[1:nAttr+1,:],shapley_CAEq_npv_priv_std[1:nAttr+1,:]]))
    max_shap = 1.1*np.max(np.vstack([shapley_CAEq_npv_mean[1:nAttr+1,:],shapley_CAEq_npv_sens_mean[1:nAttr+1,:],shapley_CAEq_npv_priv_mean[1:nAttr+1,:]])+np.vstack([shapley_CAEq_npv_std[1:nAttr+1,:],shapley_CAEq_npv_sens_std[1:nAttr+1,:],shapley_CAEq_npv_priv_std[1:nAttr+1,:]]))
    
    plt.show()
    fig = plt.figure(figsize=(5, 6))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_CAEq_npv_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_CAEq_npv_mean[ii,:] - shapley_CAEq_npv_std[ii,:], shapley_CAEq_npv_mean[ii,:] + shapley_CAEq_npv_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].plot(thresh,shapley_CAEq_npv_sens_mean[ii,:], label = 'Protected group', color='r')
        axs[ii-1].fill_between(thresh, shapley_CAEq_npv_sens_mean[ii,:] - shapley_CAEq_npv_sens_std[ii,:], shapley_CAEq_npv_sens_mean[ii,:] + shapley_CAEq_npv_sens_std[ii,:],alpha=0.3, color='r')
        axs[ii-1].plot(thresh,shapley_CAEq_npv_priv_mean[ii,:], label = 'Privileged group', color='b')
        axs[ii-1].fill_between(thresh, shapley_CAEq_npv_priv_mean[ii,:] - shapley_CAEq_npv_priv_std[ii,:], shapley_CAEq_npv_priv_mean[ii,:] + shapley_CAEq_npv_priv_std[ii,:],alpha=0.3, color='b')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(0.05, 0.20, attr_names[ii-1])
        #○axs[ii-1].set_title(attr_names[ii-1],horizontalalignment='right',verticalalignment='center')
        plt.xlabel('Threshold',fontsize=11)
        #if ii == 1:
        #    axs[ii-1].legend(loc="upper right",fontsize=12)
    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on the NPV',fontsize=11, va='center', rotation='vertical')
    fig.legend(['Overall', 'Protected group', 'Privileged group'], bbox_to_anchor =(0.45, 0.96), loc='upper center', ncol=3)
   
    plt.show()
    plt.plot(thresh, np.mean(npv[-1,:,:],axis=1), 'k', thresh, np.mean(npv_sens[-1,:,:],axis=1), 'r', thresh, np.mean(npv_priv[-1,:,:],axis=1), 'b', thresh, np.flip(thresh), 'k--')
    plt.legend(['Overall', 'Protected group', 'Privileged group', 'Random classifier'])
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('NPV',fontsize=12)

    plt.show()
    for ii in range(nAttr):
        plt.plot(thresh, shapley_CAEq_mean[ii+1,:])
    plt.legend(attr_names,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Threshold',fontsize=11)
    plt.ylabel('Conditional Use Accuracy Equality',fontsize=11)
    
    return shapley_CAEq_npv_mean, shapley_CAEq_npv_sens_mean, shapley_CAEq_npv_priv_mean, shapley_CAEq_mean, shapley_CAEq_npv_std, shapley_CAEq_npv_sens_std, shapley_CAEq_npv_priv_std, shapley_CAEq_std

def overall_accuracy_equality(tp,tn,tp_sens,tn_sens,tp_priv,tn_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold):
    " This function calculates the Shapley values for Overall Accuracy Equality "
    " It also plots the results "
    
    oae = (tp + tn)/np.tile(nSens+nPriv,[2**nAttr,len(thresh),1])
    oae_sens = (tp_sens + tn_sens)/np.tile(nSens, [2**nAttr, len(thresh),1])
    oae_priv = (tp_priv + tn_priv)/np.tile(nPriv, [2**nAttr, len(thresh),1])
    OAEq = np.abs(oae_sens - oae_priv)

    " Calculating Shapley values "
    shapley_OAEq_all, shapley_OAEq_sens, shapley_OAEq_priv, shapley_OAEq = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))
    for kk in range(nFold):
        shapley_OAEq_all[:,:,kk] = transf_matrix @ oae[:,:,kk]
        shapley_OAEq_sens[:,:,kk] = transf_matrix @ oae_sens[:,:,kk]
        shapley_OAEq_priv[:,:,kk] = transf_matrix @ oae_priv[:,:,kk]
        shapley_OAEq[:,:,kk] = transf_matrix @ OAEq[:,:,kk]
        
    shapley_OAEq_all_mean = np.mean(shapley_OAEq_all,axis=2)
    shapley_OAEq_sens_mean = np.mean(shapley_OAEq_sens,axis=2)
    shapley_OAEq_priv_mean = np.mean(shapley_OAEq_priv,axis=2)
    shapley_OAEq_mean = np.mean(shapley_OAEq,axis=2)

    shapley_OAEq_all_std = np.std(shapley_OAEq_all,axis=2)
    shapley_OAEq_sens_std = np.std(shapley_OAEq_sens,axis=2)
    shapley_OAEq_priv_std = np.std(shapley_OAEq_priv,axis=2)
    shapley_OAEq_std = np.std(shapley_OAEq,axis=2)
    
    min_shap = 1.1*np.min(np.vstack([shapley_OAEq_all_mean[1:nAttr+1,:],shapley_OAEq_sens_mean[1:nAttr+1,:],shapley_OAEq_priv_mean[1:nAttr+1,:]])-np.vstack([shapley_OAEq_all_std[1:nAttr+1,:],shapley_OAEq_sens_std[1:nAttr+1,:],shapley_OAEq_priv_std[1:nAttr+1,:]]))
    max_shap = 1.1*np.max(np.vstack([shapley_OAEq_all_mean[1:nAttr+1,:],shapley_OAEq_sens_mean[1:nAttr+1,:],shapley_OAEq_priv_mean[1:nAttr+1,:]])+np.vstack([shapley_OAEq_all_std[1:nAttr+1,:],shapley_OAEq_sens_std[1:nAttr+1,:],shapley_OAEq_priv_std[1:nAttr+1,:]]))
    
    plt.show()
    fig = plt.figure(figsize=(5, 6))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_OAEq_all_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_OAEq_all_mean[ii,:] - shapley_OAEq_all_std[ii,:], shapley_OAEq_all_mean[ii,:] + shapley_OAEq_all_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].plot(thresh,shapley_OAEq_sens_mean[ii,:], label = 'Protected group', color='r')
        axs[ii-1].fill_between(thresh, shapley_OAEq_sens_mean[ii,:] - shapley_OAEq_sens_std[ii,:], shapley_OAEq_sens_mean[ii,:] + shapley_OAEq_sens_std[ii,:],alpha=0.3, color='r')
        axs[ii-1].plot(thresh,shapley_OAEq_priv_mean[ii,:], label = 'Privileged group', color='b')
        axs[ii-1].fill_between(thresh, shapley_OAEq_priv_mean[ii,:] - shapley_OAEq_priv_std[ii,:], shapley_OAEq_priv_mean[ii,:] + shapley_OAEq_priv_std[ii,:],alpha=0.3, color='b')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(0.05, 0.20, attr_names[ii-1])
        #○axs[ii-1].set_title(attr_names[ii-1],horizontalalignment='right',verticalalignment='center')
        plt.xlabel('Threshold',fontsize=11)
        #if ii == 1:
        #    axs[ii-1].legend(loc="upper right",fontsize=12)
    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on the Overall Accuracy Equality',fontsize=11, va='center', rotation='vertical')
    fig.legend(['Overall', 'Protected group', 'Privileged group'], bbox_to_anchor =(0.45, 0.96), loc='upper center', ncol=3)
   
    plt.show()
    plt.plot(thresh, np.mean(oae[-1,:,:],axis=1), 'k', thresh, np.mean(oae_sens[-1,:,:],axis=1), 'r', thresh, np.mean(oae_priv[-1,:,:],axis=1), 'b', thresh, np.flip(thresh), 'k--')
    plt.legend(['Overall', 'Protected group', 'Privileged group', 'Random classifier'])
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('Overall accuracy equality',fontsize=12)

    plt.show()
    for ii in range(nAttr):
        plt.plot(thresh, shapley_OAEq_mean[ii+1,:])
    plt.legend(attr_names,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Threshold',fontsize=11)
    plt.ylabel('Overall Accuracy Equality',fontsize=11)
    
    return shapley_OAEq_all_mean, shapley_OAEq_sens_mean, shapley_OAEq_priv_mean, shapley_OAEq_mean, shapley_OAEq_all_std, shapley_OAEq_sens_std, shapley_OAEq_priv_std, shapley_OAEq_std

def treatment_equality(fn,fp,fn_sens,fp_sens,fn_priv,fp_priv,nSens,nPriv,nAttr,transf_matrix,thresh,attr_names,nFold):
    " This function calculates the Shapley values for Treatment Equality "
    " It also plots the results "
    
    te = fn/fp
    te[np.isnan(te)], te[np.isinf(te)] = 0, 0
    te_sens, te_priv = fn_sens/fp_sens, fn_priv//fp_priv
    te_sens[np.isnan(te_sens)], te_priv[np.isnan(te_priv)], te_sens[np.isinf(te_sens)], te_priv[np.isinf(te_priv)] = 0, 0, 0, 0
    TrEq = np.abs(te_sens - te_priv)

    " Calculating Shapley values "
    shapley_TrEq_all, shapley_TrEq_sens, shapley_TrEq_priv, shapley_TrEq = np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold)), np.zeros((2**nAttr,len(thresh),nFold))
    for kk in range(nFold):
        shapley_TrEq_all[:,:,kk] = transf_matrix @ te[:,:,kk]
        shapley_TrEq_sens[:,:,kk] = transf_matrix @ te_sens[:,:,kk]
        shapley_TrEq_priv[:,:,kk] = transf_matrix @ te_priv[:,:,kk]
        shapley_TrEq[:,:,kk] = transf_matrix @ TrEq[:,:,kk]
        
    shapley_TrEq_all_mean = np.mean(shapley_TrEq_all,axis=2)
    shapley_TrEq_sens_mean = np.mean(shapley_TrEq_sens,axis=2)
    shapley_TrEq_priv_mean = np.mean(shapley_TrEq_priv,axis=2)
    shapley_TrEq_mean = np.mean(shapley_TrEq,axis=2)

    shapley_TrEq_all_std = np.std(shapley_TrEq_all,axis=2)
    shapley_TrEq_sens_std = np.std(shapley_TrEq_sens,axis=2)
    shapley_TrEq_priv_std = np.std(shapley_TrEq_priv,axis=2)
    shapley_TrEq_std = np.std(shapley_TrEq,axis=2)
    
    min_shap = 1.1*np.min(np.vstack([shapley_TrEq_all_mean[1:nAttr+1,:],shapley_TrEq_sens_mean[1:nAttr+1,:],shapley_TrEq_priv_mean[1:nAttr+1,:]])-np.vstack([shapley_TrEq_all_std[1:nAttr+1,:],shapley_TrEq_sens_std[1:nAttr+1,:],shapley_TrEq_priv_std[1:nAttr+1,:]]))
    max_shap = 1.1*np.max(np.vstack([shapley_TrEq_all_mean[1:nAttr+1,:],shapley_TrEq_sens_mean[1:nAttr+1,:],shapley_TrEq_priv_mean[1:nAttr+1,:]])+np.vstack([shapley_TrEq_all_std[1:nAttr+1,:],shapley_TrEq_sens_std[1:nAttr+1,:],shapley_TrEq_priv_std[1:nAttr+1,:]]))
    
    plt.show()
    fig = plt.figure(figsize=(5, 6))
    gs = fig.add_gridspec(nAttr, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    for ii in range(1,nAttr+1):
        axs[ii-1].plot(thresh,shapley_TrEq_all_mean[ii,:], label = 'Overall', color='k')
        axs[ii-1].fill_between(thresh, shapley_TrEq_all_mean[ii,:] - shapley_TrEq_all_std[ii,:], shapley_TrEq_all_mean[ii,:] + shapley_TrEq_all_std[ii,:],alpha=0.3, color='k')
        axs[ii-1].plot(thresh,shapley_TrEq_sens_mean[ii,:], label = 'Protected group', color='r')
        axs[ii-1].fill_between(thresh, shapley_TrEq_sens_mean[ii,:] - shapley_TrEq_sens_std[ii,:], shapley_TrEq_sens_mean[ii,:] + shapley_TrEq_sens_std[ii,:],alpha=0.3, color='r')
        axs[ii-1].plot(thresh,shapley_TrEq_priv_mean[ii,:], label = 'Privileged group', color='b')
        axs[ii-1].fill_between(thresh, shapley_TrEq_priv_mean[ii,:] - shapley_TrEq_priv_std[ii,:], shapley_TrEq_priv_mean[ii,:] + shapley_TrEq_priv_std[ii,:],alpha=0.3, color='b')
        axs[ii-1].set_ylim([min_shap, max_shap])
        axs[ii-1].text(0.05, 0.20, attr_names[ii-1])
        #○axs[ii-1].set_title(attr_names[ii-1],horizontalalignment='right',verticalalignment='center')
        plt.xlabel('Threshold',fontsize=11)
        #if ii == 1:
        #    axs[ii-1].legend(loc="upper right",fontsize=12)
    for ax in axs:
        ax.label_outer()
    # set labels
    plt.setp(axs[-1], xlabel='Threshold')
    fig.text(-0.015, 0.5, 'Feature contribution on the Treatment Equality',fontsize=11, va='center', rotation='vertical')
    fig.legend(['Overall', 'Protected group', 'Privileged group'], bbox_to_anchor =(0.45, 0.96), loc='upper center', ncol=3)
   
    plt.show()
    plt.plot(thresh, np.mean(te[-1,:,:],axis=1), 'k', thresh, np.mean(te_sens[-1,:,:],axis=1), 'r', thresh, np.mean(te_priv[-1,:,:],axis=1), 'b', thresh, np.flip(thresh), 'k--')
    plt.legend(['Overall', 'Protected group', 'Privileged group', 'Random classifier'])
    plt.xlabel('Threshold',fontsize=12)
    plt.ylabel('Treatment Equality',fontsize=12)

    plt.show()
    for ii in range(nAttr):
        plt.plot(thresh, shapley_TrEq_mean[ii+1,:])
    plt.legend(attr_names,fontsize=12,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Threshold',fontsize=11)
    plt.ylabel('Treatment Equality',fontsize=11)
    
    return shapley_TrEq_all_mean, shapley_TrEq_sens_mean, shapley_TrEq_priv_mean, shapley_TrEq_mean, shapley_TrEq_all_std, shapley_TrEq_sens_std, shapley_TrEq_priv_std, shapley_TrEq_std


def plot_waterfall(nAttr,values,values_std,names,y_label):
    " Waterfall plots "
    values_argsort =  np.abs(values[1:]).argsort()[::-1]
    values_sort =  np.hstack([values[0],values[values_argsort+1]])
    increment = np.zeros((nAttr+2,))
    increment[0:nAttr+1] = values_sort
    increment[-1] = sum(increment)
    start_point = np.zeros((len(increment)))
    position = np.zeros((nAttr+2,))
    position[0] = increment[0]
    position[-1] = sum(increment[0:-1])
    for ii in range(len(increment)-2):
        start_point[ii+1] = start_point[ii] + increment[ii]
        position[ii+1] = position[ii] + increment[ii+1]

    increment, start_point, position, values, values_std = increment, start_point, position, values, values_std
    attr_names_all = names[values_argsort].insert(0,'Rand. Class.')
    attr_names_all = attr_names_all.insert(len(attr_names_all),y_label)

    colors_bar = ["black"]
    for ii in increment[1:-1]:
        if ii >= 0:
            colors_bar.append("green")
        else:
            colors_bar.append("red")
    colors_bar.append("blue")
    
    #values_std = 0

    width = 0.75
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    bar_plot = ax.bar(attr_names_all, increment, width,bottom=start_point, color=colors_bar, yerr=values_std, edgecolor = 'black', capsize = 7)
    plt.xticks(rotation=90, fontsize=13)
    ax.set_ylim([-0.001, max(0.05, max(position)+0.015)])
    ax.set_ylabel('Feature contribution on the {}'.format(y_label),fontsize=13)
    ii = 0
    for rect in bar_plot:
        if ii == 0:
            plt.text(rect.get_x() + rect.get_width()/15., 0.001+position[ii],'%.2f' % increment[ii],ha='center', va='bottom')
        else:
            plt.text(rect.get_x() + rect.get_width()/15., 0.001+max(position[ii],position[ii-1]),'%.2f' % increment[ii],ha='center', va='bottom')
        ii += 1
    plt.show()
