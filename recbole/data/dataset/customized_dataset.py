# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""

import numpy as np
import torch

from recbole.data.dataset import KGSeqDataset, SequentialDataset
from recbole.data.interaction import Interaction
from recbole.sampler import SeqSampler
from recbole.utils.enum_type import FeatureType
from recbole.data.dataset import Dataset
from recbole.utils import FeatureSource, FeatureType, get_local_time, set_color
import pandas as pd
from scipy.sparse import coo_matrix
from collections import Counter, defaultdict

class TagBasedDataset(Dataset):
    """:class:`TagBasedDataset` is based on :`~recbole.data.dataset.dataset.Dataset`,
    and load_col 'tag_id' additionally

    tag assisgment [``user_id``, ``item_id``, ``tag_id``]

    Attributes:
        tid_field (str): The same as ``config['TAG_ID_FIELD']``

    """

    def __init__(self, config):
        super().__init__(config)

    def _get_field_from_config(self):
        super()._get_field_from_config()

        self.tid_field = self.config['TAG_ID_FIELD']
        self._check_field('tid_field')
        self.set_field_property(self.tid_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        if self.tid_field is None:
            raise ValueError(
                'Tag need to be set at the same time or not set at the same time.'
            )
        self.logger.debug(set_color('tid_field', 'blue') + f': {self.tid_field}')

    def _data_filtering(self):
         super()._data_filtering()
         # self._filter_tag()

    # def _filter_tag(self):
    #     pass

    #def _load_data(self, token, dataset_path):
    #    super()._load_data(token, dataset_path)

    # def _build_feat_name_list(self):
    #     feat_name_list = super()._build_feat_name_list()

    def _data_processing(self):
        """Data preprocessing, including:

        - Data filtering
        - Remap ID
        - Missing value imputation
        - Normalization
        - Preloading weights initialization
        """
        super()._data_processing()
        # self.user_tag_value_matrix = self.create_src_tgt_matrix(self.inter_feat, self.uid_field, self.tid_field)
        # self.item_tag_value_matrix = self.create_src_tgt_matrix(self.inter_feat, self.iid_field, self.tid_field)
        # self.user_tag_matrix = self._create_ui_tag_matrix(self.inter_feat, self.uid_field, self.tid_field, form='coo', is_weight=False)
        # self.item_tag_matrix = self._create_ui_tag_matrix(self.inter_feat, self.iid_field, self.tid_field, form='coo', is_weight=False)

    def __str__(self):
        info = [
            super().__str__()
        ]
        if self.tid_field:
            info.extend([
                set_color('The number of tags', 'blue') + f': {self.tag_num}',
                set_color('Average actions of tags', 'blue') + f': {self.avg_actions_of_tags}'
            ])
        return '\n'.join(info)

    # def _build_feat_name_list(self):
    #     feat_name_list = super()._build_feat_name_list()
    #     if self.tid_field is not None:
    #         feat_name_list.append('tag_feat')
    #     return feat_name_list

    def _init_alias(self):
        """Add :attr:`alias_of_tag_id` and update :attr:`_rest_fields`.
        """
        self._set_alias('tag_id', [self.tid_field])
        super()._init_alias()

    def create_src_tgt_matrix(self, df_feat, source_field, target_field, is_weight=True):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            df_feat (Interaction): Feature where src and tgt exist.
            source_field (str): Source field
            target_field (str): Target field
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not isinstance(df_feat, pd.DataFrame):
            try:
                df_feat = pd.DataFrame.from_dict(df_feat.interaction)
            except BaseException:
                raise ValueError(f'feat from is not supported.')
        df_feat = df_feat.groupby([source_field, target_field]).size()
        df_feat.name = 'weights'
        df_feat = df_feat.reset_index()
        src = df_feat[source_field]
        tgt = df_feat[target_field]
        if is_weight:
            data = df_feat['weights']
        else:
            data = np.ones(len(df_feat))
        mat = coo_matrix((data, (src, tgt)), shape=(self.num(source_field), self.num(target_field)))
        return mat
        # if form == 'coo':
        #     return mat
        # elif form == 'csr':
        #     return mat.tocsr()
        # else:
        #     raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')

    def inter_graph(self, form='dgl', value_field=None):

        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')
        return self._create_graph(self.inter_feat, self.uid_field, self.iid_field, form, value_field)

    @property
    def tag_num(self):
        """Get the number of different tokens of tags.

       Returns:
           int: Number of different tokens of tags.
       """
        return self.num(self.tid_field)

    @property
    def avg_actions_of_tags(self):
        """Get the average number of tags' interaction records.

        Returns:
             numpy.float64: Average number of tags' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.tid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.tid_field].numpy()).values()))

class GRU4RecKGDataset(KGSeqDataset):

    def __init__(self, config):
        super().__init__(config)


class KSRDataset(KGSeqDataset):

    def __init__(self, config):
        super().__init__(config)


class DIENDataset(SequentialDataset):
    """:class:`DIENDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It add users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored DIENDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def __init__(self, config):
        super().__init__(config)

        list_suffix = config['LIST_SUFFIX']
        neg_prefix = config['NEG_PREFIX']
        self.seq_sampler = SeqSampler(self)
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix
        self.neg_item_list = self.seq_sampler.sample_neg_sequence(self.inter_feat[self.iid_field])

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug('data_augmentation')

        self._aug_presets()

        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f'{field}_list_field')
                list_len = self.field2seqlen[list_field]
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                list_ftype = self.field2type[list_field]
                dtype = torch.int64 if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ] else torch.float64
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]

                # DIEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                        new_dict[self.neg_item_list_field][i][:length] = self.neg_item_list[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data
