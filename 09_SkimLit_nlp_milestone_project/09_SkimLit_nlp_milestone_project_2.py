#!/usr/bin/env python
# coding: utf-8

# # Milestone Poject 2 : skimLit
# 
# The purpose of this notebook is to build an NLP model to make reading medical abstracts easier.
# 
# The paper we're replicating (the source of the dataset that we'll be using) is available here : https://arxiv.org/abs/1710.06071
#         
# And reading through the paper above, we see that the model architecture that they use to achieve their best results is available : https://arxiv.org/abs/1612.05251
#         
# *Resource :* If you want to find the ground truth for this notebook(with lots of diagrams and text annotation) see the Github: https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/09_SkimLit_nlp_milestone_project_2.ipynb

# In[34]:


# Confirm acces to GPU
get_ipython().system('nvidia-smi')


# ## Get data
# 
# Since we'll be replicating the paper above (PubMed 200K RCT)? let's download the dataset they used,
# We can do so from the authors GitHub : 

# In[42]:


get_ipython().system('git clone https://github.com/Franck-Dernoncourt/pubmed-rct')
# Check what files are in the PubMed_20K dataset


# In[43]:


ls pubmed-rct


# In[49]:


ls pubmed-rct\PubMed_20k_RCT


# In[46]:


ls pubmed-rct\PubMed_20k_RCT_numbers_replaced_with_at_sign


# In[ ]:




