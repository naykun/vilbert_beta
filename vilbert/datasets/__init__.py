from .concept_cap_dataset import ConceptCapLoaderTrain, ConceptCapLoaderVal, ConceptCapLoaderRetrieval
from .vqa_dataset import VQAClassificationDataset
from .refer_expression_dataset import ReferExpressionDataset
from .retreival_dataset import RetreivalDataset, RetreivalDatasetVal
from .vcr_dataset import VCRDataset
from .vgsg_dataset import VGSGDataset
from .vgsg_novision_dataset import VGSGNoVisDataset



# from .flickr_retreival_dataset import FlickrRetreivalDatasetTrain, FlickrRetreivalDatasetVal

__all__ = [
		   "VQAClassificationDataset", \
		   "ConceptCapLoaderTrain", \
		   "ConceptCapLoaderVal", \
		   "ReferExpressionDataset", \
		   "RetreivalDataset", \
		   "RetreivalDatasetVal",\
		   "VCRDataset", \
		   "ConceptCapLoaderRetrieval",\
		   "VGSGDataset",\
			"VGSGNoVisDataset"]

DatasetMapTrain = {
				   'TASK0': VQAClassificationDataset,
				   'TASK1': VCRDataset,
				   'TASK2': VCRDataset,				   
				   'TASK3': RetreivalDataset,
				   'TASK4': ReferExpressionDataset,
				   'TASK5': VGSGDataset,
				   'TASK6': VGSGNoVisDataset,
				   }		

DatasetMapEval = {
				 'TASK0': VQAClassificationDataset,
				 'TASK1': VCRDataset,
				 'TASK2': VCRDataset,				   
				 'TASK3': RetreivalDatasetVal,
				 'TASK4': ReferExpressionDataset,
				 'TASK5': VGSGDataset,
				 'TASK6': VGSGNoVisDataset,			   
				}
