
import torch
import torch.nn.functional as F
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from TSPLocalSearch import TSPLocalSearch as LocalSearch

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

EPS = 1e-6

class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 search_params=None):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.search_params = search_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        if search_params:
            self.local_search = LocalSearch(**self.search_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Loss Components
        self.loss_type = trainer_params['loss_type']
        self.alpha = trainer_params['alpha']

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        route_mean_log_prob_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        step = 0
        while episode < train_num_episode:
            step = step + 1
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size, step)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                        .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                                score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size, step):
        # Augmentation
        ###############################################
        if self.trainer_params['augmentation_enable']:
            aug_factor = self.trainer_params['aug_factor']
        else:
            aug_factor = 1

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size // aug_factor, aug_factor=aug_factor)

        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        # Loss
        ###############################################
        if self.loss_type == 'po_ls_loss':
            loss = self.preference_among_two_opt_loss_fn(reward, prob_list)
        elif self.loss_type == 'po_loss':
            loss = self.preference_among_pomo_loss_fn(reward, prob_list)
        elif self.loss_type == 'pl_loss':
            loss = self.rank_among_pomo_loss_fn(reward, prob_list)
        elif self.loss_type == 'rl_loss':
            loss = self.rl_loss_fn(reward, prob_list)
        else:
            raise NotImplementedError
        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        loss = loss / self.trainer_params['optimizer_step_interval']
        loss.backward()
        if step % self.trainer_params['optimizer_step_interval'] == 0:
            self.optimizer.step()
            self.model.zero_grad()
        return score_mean.item(), loss.item()

    def rl_loss_fn(self, reward, prob_list):
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        return loss_mean
    
    def kl_loss_fn(self, prob, reference_prob):
        log_prob = torch.log(prob)
        reference_log_prob = torch.log(reference_prob)
        return torch.mean(prob * (log_prob - reference_log_prob))
    
    def preference_among_pomo_loss_fn(self, reward, prob): # BT
        preference = reward[:, :, None] > reward[:, None, :]
        # shape: (batch, pomo, pomo)

        log_prob = torch.log(prob).sum(2)
        log_prob_pair = log_prob[:, :, None] - log_prob[:, None, :]
        pf_log = torch.log(F.sigmoid(self.alpha * log_prob_pair))
        loss = -torch.mean(pf_log * preference)
        
        return loss
    
    def rank_among_pomo_loss_fn(self, reward, prob): # PL
        sorted_index = torch.sort(reward, dim=1, descending=True)[1]
        # shape: (batch, pomo)
        log_prob = self.alpha * torch.log(prob).sum(2)
        max_log_prob = log_prob.max(1, keepdim=True)[0]
        log_prob = log_prob - max_log_prob
        exp_log_prob = torch.exp(log_prob)
        one_hot = F.one_hot(sorted_index).to(torch.float)
        # shape: (batch, pomo, pomo)
        till_mat = torch.tril(torch.ones_like(one_hot))
        sum_exp = (till_mat @ one_hot @ exp_log_prob[:, :, None]).squeeze(-1)
        # shape: (batch, pomo)
        loss = torch.mean(torch.log(exp_log_prob) - torch.log(sum_exp))

        return loss

    def preference_among_two_opt_loss_fn(self, reward, probs):
        dist = self.env.get_distmat()
        route_info = self.local_search.search(self.env.selected_node_list, reward, dist, self.env.problems)
        
        search_probs = self.model.route_forward(route_info) + EPS
        probs = torch.cat((probs, search_probs), dim=1)
        reward = torch.cat((reward, route_info.reward), dim=1)
        return self.preference_among_pomo_loss_fn(reward, probs)