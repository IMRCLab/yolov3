import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import os

# Script to plot predicted vs. ground-truth positions

class PlotPosition:
    figure_counter = 0
    
    @staticmethod
    def plot (gt_x, gt_y, gt_z, predicted_x, predicted_y, predicted_z):
        images = range(1, len(predicted_x) + 1)

        PlotPosition.figure_counter += 1
        plt.figure(PlotPosition.figure_counter, figsize=(20, 12))
        plt.margins(0.1)

        gs = gridspec.GridSpec(1, 3)
        ax = plt.subplot(gs[0, 0])
        ax.set_title('x')

        plt.plot(images, gt_x, color='black', label='Ground truth x ')
        plt.plot(images, predicted_x, color='green', label='Predicted x')
        plt.legend()

        ax = plt.subplot(gs[0, 1])
        ax.set_title('y')

        plt.plot(images, gt_y, color='black', label='Ground truth y')
        plt.plot(images, predicted_y, color='blue', label='Predicted y')
        plt.legend()

        ax = plt.subplot(gs[0, 2])
        ax.set_title('z')

        plt.plot(images, gt_z, color='black', label='Ground truth z')
        plt.plot(images, predicted_z, color='r', label='Predicted z')
        plt.legend()


        plt.subplots_adjust(hspace=0.3)
        plt.suptitle('Ground truth vs. Prediction (in mm.)')
        plt.xlabel('Images')
        plt.ylabel('')

        plt.show()
        # fig_path = os.path.join(PlotPosition.folderPath, '{}Compare_pred_gt.png'.format(PlotPosition.desc))
        # plt.savefig('gt-vs-prediction-25.jpg')

class PlotMSELoss:
    figure_counter = 0
    @staticmethod
    def plot(train_losses_x, train_losses_y, train_losses_z):
        images = range(1, len(train_losses_x) + 1)
        PlotMSELoss.figure_counter += 1
        plt.figure(PlotMSELoss.figure_counter, figsize=(20, 12))
        plt.margins(0.1)

        gs = gridspec.GridSpec(1, 3)
        ax = plt.subplot(gs[0, 0])
        ax.set_title('x')

        plt.plot(images, train_losses_x, color='green', label='squarred error x')
        plt.legend()

        ax = plt.subplot(gs[0, 1])
        ax.set_title('y')

        plt.plot(images, train_losses_y, color='blue', label='squarred error y')
        plt.legend()

        ax = plt.subplot(gs[0, 2])
        ax.set_title('z')

        plt.plot(images, train_losses_z, color='r', label='squarred error z')
        plt.legend()

        plt.subplots_adjust(hspace=0.3)
        plt.suptitle('Test error in cm.')
        # plt.xlabel('images')
        # plt.ylabel('error')

        # fig_path = os.path.join(ErrorVisualization.folderPath, '{}synthetic4k_RMSE_mae_trained.png'.format(ErrorVisualization.desc))
        # plt.savefig(fig_path)
        plt.show()

class PlotHist:
    def plot(eucl_error, name):
        plt.hist(eucl_error)
        plt.suptitle('Histogram for Eucl. in m. with yolo')
        plt.savefig(name)
        plt.show()

class PlotWhisker:
    def plot(eucl_error, name, title_name):
        fig = plt.figure(figsize =(10, 7))
        plt.title(title_name)
        plt.boxplot(eucl_error)
        plt.savefig(name)
        # plt.show()