import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import wandb
from configs.train_config import CFG


def calculate_metrics(target, output, average = CFG.metrics_avg, conf_matrix = False):

    class_lst = ['normal nail (0)', 'onychomycosis  (1)', 'nail dystrophy  (2)', 'onycholysis  (3)', 'melanonychia  (4)']
    accuracy = accuracy_score(target, output)
    recall = recall_score(target, output, average=average)
    precision = precision_score(target, output, average=average)
    f1 = f1_score(target, output, average=average)
    if conf_matrix:
        cm = confusion_matrix(target, output)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        fig.colorbar(cax)
        ax.set_yticklabels([''] + class_lst)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig(CFG.output_path+'/confusion_matrix.png')
        plt.show()        

    return accuracy, recall, precision, f1


def print_loss_and_metrics(target, output, class_dict, fold=None, epoch=None, train_loss=None, val_loss=None):
    
    if epoch == 1:
        print(f"\nFold {fold}:\n")
    elif epoch is None:
        print("\nAll folds:\n")

    ##### metrics per class #####
    _, recall_class, precision_class, f1_class = calculate_metrics(target, output, average=None, conf_matrix=False)
    for i in range(CFG.num_classes - 1):
        print(f"{class_dict[i]}........ Recall: {recall_class[i]:.4f}, Precision: {precision_class[i]:.4f}, F1: {f1_class[i]:.4f}")
    
    ##### normal nail vs. nail with disease #####
    target_disease = target > 0
    output_disease = output > 0
    
    accuracy_disease, recall_disease, precision_disease, f1_disease = calculate_metrics(target_disease, output_disease, \
                                                                                        average='binary', conf_matrix=False)
    
    ##### average metrics #####
    accuracy, recall, precision, f1 = calculate_metrics(target, output, average=CFG.metrics_avg, conf_matrix=True)
    
    print(f"\nNail with disease vs. healthy nail: Accuracy val: {accuracy_disease:.4f} - Recall val: {recall_disease:.4f} " +
          f"- Precision val: {precision_disease:.4f} - F1 val: {f1_disease:.4f}\n") 
    
    if train_loss is None:
        print(f"All folds: .............. Accuracy val: {accuracy:.4f} - Recall val: {recall:.4f} - Precision val: {precision:.4f} " +
          f"- F1 val: {f1:.4f}\n")
        
        if CFG.wandb:
            wandb.log({"Accuracy": accuracy,
                       "Recall": recall,
                       "Precision": precision,
                       "F1 score": f1})
    else:
        print(f"Epoch: {epoch}............. Loss: {train_loss.avg:.4f} - Loss val: {val_loss.avg:.4f} " +
          f"- Accuracy val: {accuracy:.4f} - Recall val: {recall:.4f} - Precision val: {precision:.4f} " +
          f"- F1 val: {f1:.4f}\n")
        
        if CFG.wandb:
            wandb.log({f"[fold {fold}]: Loss train": train_loss.avg,
                       f"[fold {fold}]: Loss val": val_loss.avg,
                       f"[fold {fold}]: Accuracy": accuracy,
                       f"[fold {fold}]: Recall": recall,
                       f"[fold {fold}]: Precision": precision,
                       f"[fold {fold}]: F1 score": f1})
    return f1