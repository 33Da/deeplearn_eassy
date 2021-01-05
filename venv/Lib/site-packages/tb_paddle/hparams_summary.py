# coding=utf-8
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import hashlib
import json
import random
import six
import time

from .proto import summary_pb2
from .proto import api_pb2
from .proto import plugin_data_pb2


@six.add_metaclass(abc.ABCMeta)
class Domain(object):
    """The domain of a hyperparameter.
    """

    @abc.abstractproperty
    def dtype(self):
        """Data type of this domain: `float`, `int`, `str`, or `bool`."""
        pass

    @abc.abstractmethod
    def sample_uniform(self, rng=random):
        """Sample a value from this domain uniformly at random.

        :param rng: A `random.Random` interface; defaults to the `random` module itself.

        Raises:
          IndexError: If the domain is empty.
        """
        pass

    @abc.abstractmethod
    def update_hparam_info(self, hparam_info):
        """Update an `HParamInfo` proto to include this domain.

        This should update the `type` field on the proto and exactly one of
        the `domain` variants on the proto.
        
        :param  hparam_info: An `api_pb2.HParamInfo` proto to modify.
        """
        pass


class Discrete(Domain):
    """A domain that takes on a fixed set of values.

    These values may be of any (single) domain type.
    """

    def __init__(self, values, dtype=None):
        """Construct a discrete domain.

        :param values: A iterable of the values in this domain.
        :type values: list, numpy.array
        :param dtype: The Python data type of values in this domain: one of
            `int`, `float`, `bool`, or `str`. If `values` is non-empty,
            `dtype` may be `None`, in which case it will be inferred as the
            type of the first element of `values`.

        Raises:
          ValueError: If `values` is empty but no `dtype` is specified.
          ValueError: If `dtype` or its inferred value is not `int`, `float`, `bool`, or `str`.
          TypeError: If an element of `values` is not an instance of `dtype`.
        """
        self._values = list(values)
        if dtype is None:
            if self._values:
                dtype = type(self._values[0])
            else:
                raise ValueError("Empty domain with no dtype specified")
        if dtype not in (int, float, bool, str):
            raise ValueError("Unknown dtype: %r" % (dtype,))
        self._dtype = dtype
        for value in self._values:
            if not isinstance(value, self._dtype):
                raise TypeError(
                    "dtype mismatch: not isinstance(%r, %s)"
                    % (value, self._dtype.__name__)
                )
        self._values.sort()

    def __str__(self):
        return "{%s}" % (", ".join(repr(x) for x in self._values))

    def __repr__(self):
        return "Discrete(%r)" % (self._values,)

    @property
    def dtype(self):
        return self._dtype

    @property
    def values(self):
        return list(self._values)

    def sample_uniform(self, rng=random):
        return rng.choice(self._values)

    def update_hparam_info(self, hparam_info):
        hparam_info.type = {
            int: api_pb2.DATA_TYPE_FLOAT64,
            float: api_pb2.DATA_TYPE_FLOAT64,
            bool: api_pb2.DATA_TYPE_BOOL,
            str: api_pb2.DATA_TYPE_STRING,
        }[self._dtype]
        hparam_info.ClearField("domain_discrete")
        hparam_info.domain_discrete.extend(self._values)


class IntInterval(Domain):
    """A domain that takes on all integer values in a closed interval.
    """
    def __init__(self, min_value=None, max_value=None):
        """Create an `IntInterval`.

        :param min_value: The lower bound (inclusive) of the interval.
        :type min_value: int
        :param max_value: The upper bound (inclusive) of the interval.
        :type max_value: int

        Raises:
          TypeError: If `min_value` or `max_value` is not an `int`.
          ValueError: If `min_value > max_value`.
        """
        if not isinstance(min_value, int):
            raise TypeError("min_value must be an int: %r" % (min_value,))
        if not isinstance(max_value, int):
            raise TypeError("max_value must be an int: %r" % (max_value,))
        if min_value > max_value:
            raise ValueError("%r > %r" % (min_value, max_value))
        self._min_value = min_value
        self._max_value = max_value

    def __str__(self):
        return "[%s, %s]" % (self._min_value, self._max_value)

    def __repr__(self):
        return "IntInterval(%r, %r)" % (self._min_value, self._max_value)

    @property
    def dtype(self):
        return int

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    def sample_uniform(self, rng=random):
        return rng.randint(self._min_value, self._max_value)

    def update_hparam_info(self, hparam_info):
        hparam_info.type = api_pb2.DATA_TYPE_FLOAT64
        hparam_info.domain_interval.min_value = self._min_value
        hparam_info.domain_interval.max_value = self._max_value


class RealInterval(Domain):
    """A domain that takes on all real values in a closed interval.
    """

    def __init__(self, min_value=None, max_value=None):
        """Create a `RealInterval`.

        :param min_value: The lower bound (inclusive) of the interval.
        :type min_value: float
        :param max_value: The upper bound (inclusive) of the interval.
        :type max_value: float

        Raises:
          TypeError: If `min_value` or `max_value` is not an `float`.
          ValueError: If `min_value > max_value`.
        """
        if not isinstance(min_value, float):
            raise TypeError("min_value must be a float: %r" % (min_value,))
        if not isinstance(max_value, float):
            raise TypeError("max_value must be a float: %r" % (max_value,))
        if min_value > max_value:
            raise ValueError("%r > %r" % (min_value, max_value))
        self._min_value = min_value
        self._max_value = max_value

    def __str__(self):
        return "[%s, %s]" % (self._min_value, self._max_value)

    def __repr__(self):
        return "RealInterval(%r, %r)" % (self._min_value, self._max_value)

    @property
    def dtype(self):
        return float

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    def sample_uniform(self, rng=random):
        return rng.uniform(self._min_value, self._max_value)

    def update_hparam_info(self, hparam_info):
        hparam_info.type = api_pb2.DATA_TYPE_FLOAT64
        hparam_info.domain_interval.min_value = self._min_value
        hparam_info.domain_interval.max_value = self._max_value


class HParam(object):
    """A hyperparameter in an experiment.
      
    This class describes a hyperparameter in the abstract. It ranges
    over a domain of values, but is not bound to any particular value.
    """
     
    def __init__(self, name, domain=None, display_name=None, description=None):
        """Create a hyperparameter object.

        :param name: A string ID for this hyperparameter, 
            which should be unique within an experiment.
        :type name: str
        :param domain: Describing the values that this hyperparameter can take on.
        :type domain: <class `Domain`> object, optional
        :param display_name: Human-readable display name.
        :type display_name: str, optional
        :param description: Describing this hyperparameter, to appear in TensorBoard.
        :type description: str in Markdown format, optional

        Raises:
          ValueError: If `domain` is not a `Domain`.
        """
        self._name = name
        self._domain = domain
        self._display_name = display_name
        self._description = description
        if not isinstance(self._domain, (Domain, type(None))):
            raise ValueError("not a domain: %r" % (self._domain,))

    def __str__(self):
        return "<HParam %r: %s>" % (self._name, self._domain)

    def __repr__(self):
        fields = [
            ("name", self._name),
            ("domain", self._domain),
            ("display_name", self._display_name),
            ("description", self._description),
        ]
        fields_string = ", ".join("%s=%r" % (k, v) for (k, v) in fields)
        return "HParam(%s)" % fields_string

    @property
    def name(self):
        return self._name

    @property
    def domain(self):
        return self._domain

    @property
    def display_name(self):
        return self._display_name

    @property
    def description(self):
        return self._description


class Metric(object):
    """A metric in an experiment.
     
    A metric is a real-valued function of a model. Each metric is
    associated with a TensorBoard scalar summary, which logs the
    metric's value as the model trains.
    """
     
    TRAINING = api_pb2.DATASET_TRAINING
    VALIDATION = api_pb2.DATASET_VALIDATION
     
    def __init__(self, tag, group=None, display_name=None, description=None, dataset_type=None):
        """
        :param tag: The tag name of the scalar summary that corresponds to this metric.
        :type tag: str
        :param group: An optional string listing the subdirectory under the
            session's log directory containing summaries for this metric.
            For instance, if summaries for training runs are written to
            events files in `ROOT_LOGDIR/SESSION_ID/train`, then `group`
            should be `"train"`. Defaults to the empty string: i.e.,
            summaries are expected to be written to the session logdir.
        :param display_name: An optional human-readable display name.
        :type display_name: str
        :param description: An optional Markdown string with a human-readable
            description of this metric, to appear in TensorBoard.
        :param dataset_type: Either `Metric.TRAINING` or `Metric.VALIDATION`, or `None`.
        """
        self._tag = tag
        self._group = group
        self._display_name = display_name
        self._description = description
        self._dataset_type = dataset_type
        if self._dataset_type not in (None, Metric.TRAINING, Metric.VALIDATION):
            raise ValueError("invalid dataset type: %r" % (self._dataset_type,))
     
    def as_proto(self):
        return api_pb2.MetricInfo(
            name=api_pb2.MetricName(group=self._group, tag=self._tag,),
            display_name=self._display_name,
            description=self._description,
            dataset_type=self._dataset_type,
        )


def create_summary_metadata(hparams_plugin_data_pb):
    """Returns a summary metadata for the HParams plugin.

    Returns a summary_pb2.SummaryMetadata holding a copy of the given
    HParamsPluginData message in its plugin_data.content field.
    Sets the version field of the hparams_plugin_data_pb copy to
    PLUGIN_DATA_VERSION.

    :param hparams_plugin_data_pb: the HParamsPluginData protobuffer to use.
    """
    PLUGIN_NAME = "hparams"
    PLUGIN_DATA_VERSION = 0

    if not isinstance(hparams_plugin_data_pb, plugin_data_pb2.HParamsPluginData):
        raise TypeError(
            "Needed an instance of plugin_data_pb2.HParamsPluginData."
            " Got: %s" % type(hparams_plugin_data_pb)
        )

    content = plugin_data_pb2.HParamsPluginData()
    content.CopyFrom(hparams_plugin_data_pb)
    content.version = PLUGIN_DATA_VERSION

    return summary_pb2.SummaryMetadata(
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )


def _summary_pb(tag, hparams_plugin_data):
    """Create a summary holding the given `HParamsPluginData` message.

    :param tag: The `str` tag to use.
    :param hparams_plugin_data: The `HParamsPluginData` message to use.

    Returns: A TensorBoard `summary_pb2.Summary` message.
    """
    summary_metadata = create_summary_metadata(hparams_plugin_data)
    return summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(tag=tag, metadata=summary_metadata)]
        )

def _derive_session_group_name(trial_id, hparams):
    if trial_id is not None:
        if not isinstance(trial_id, six.string_types):
            raise TypeError(
                "`trial_id` should be a `str`, but got: %r" % (trial_id,)
            )
        return trial_id
    # Use `json.dumps` rather than `str` to ensure invariance under string
    # type (incl. across Python versions) and dict iteration order.
    jparams = json.dumps(hparams, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(jparams.encode("utf-8")).hexdigest()


def hparams_pb(hparams, trial_id=None, start_time_secs=None):
    """Create a summary encoding hyperparameter values for a single trial.

    :param hparams: A `dict` mapping hyperparameters to the values used in this
        trial. Keys should be the names of `HParam` objects usedplugin_data_pb2 in an
        experiment, or the `HParam` objects themselves. Values should be
        Python `bool`, `int`, `float`, or `string` values, depending on
        the type of the hyperparameter.
    :param trial_id: An optional `str` ID for the set of hyperparameter values
        used in this trial. Defaults to a hash of the hyperparameters.
    :param start_time_secs: The time that this trial started training, as
        seconds since epoch. Defaults to the current time.

    Returns:
        A TensorBoard `summary_pb2.Summary` message.
    """
    if start_time_secs is None:
        start_time_secs = time.time()
    result = {}
    for (k, v) in six.iteritems(hparams):
        if isinstance(k, HParam):
            k = k.name
        if k in result:
            raise ValueError("multiple values specified for hparam %r" % (k,))
        result[k] = v
    hparams = result

    group_name = _derive_session_group_name(trial_id, hparams)

    session_start_info = plugin_data_pb2.SessionStartInfo(
        group_name=group_name, start_time_secs=start_time_secs,
    )

    for hp_name in sorted(hparams):
        hp_value = hparams[hp_name]
        if isinstance(hp_value, bool):
            session_start_info.hparams[hp_name].bool_value = hp_value
        elif isinstance(hp_value, (float, int)):
            session_start_info.hparams[hp_name].number_value = hp_value
        elif isinstance(hp_value, six.string_types):
            session_start_info.hparams[hp_name].string_value = hp_value
        else:
            raise TypeError(
                "hparams[%r] = %r, of unsupported type %r"
                % (hp_name, hp_value, type(hp_value))
            )

    SESSION_START_INFO_TAG = "_hparams_/session_start_info"
    return _summary_pb(
        SESSION_START_INFO_TAG,
        plugin_data_pb2.HParamsPluginData(
            session_start_info=session_start_info
        ),
    )


def hparams_config_pb(hparams, metrics, time_created_secs=None):
    """Create a top-level experiment configuration.

    This configuration describes the hyperparameters and metrics that will
    be tracked in the experiment, but does not record any actual values of
    those hyperparameters and metrics. It can be created before any models
    are actually trained.

    :param hparams: A list of `HParam` values.
    :param metrics: A list of `Metric` values.
    :param time_created_secs: The time that this experiment was created, as
        seconds since epoch. Defaults to the current time.

    Returns:
        A TensorBoard `summary_pb2.Summary` message.
    """
    hparam_infos = []
    for hparam in hparams:
        info = api_pb2.HParamInfo(
            name=hparam.name,
            description=hparam.description,
            display_name=hparam.display_name,
        )
        domain = hparam.domain
        if domain is not None:
            domain.update_hparam_info(info)
        hparam_infos.append(info)
    metric_infos = [metric.as_proto() for metric in metrics]
    experiment = api_pb2.Experiment(
        hparam_infos=hparam_infos,
        metric_infos=metric_infos,
        time_created_secs=time_created_secs,
    )

    EXPERIMENT_TAG = "_hparams_/experiment"
    return _summary_pb(
        EXPERIMENT_TAG,
        plugin_data_pb2.HParamsPluginData(experiment=experiment),
    )


