from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from .event_file_writer import EventFileWriter
from .proto import event_pb2


class FileWriter(object):
    """Writes protocol buffers to event files to be consumed by TensorBoard.

    The `FileWriter` class provides a mechanism to create an event file in a
    given directory and add summaries and events to it. The class updates the
    file contents asynchronously.
    """

    def __init__(self, logdir, max_queue=1024, filename_suffix=''):
        """Creates a `FileWriter` and an event file.
        On construction the writer creates a new event file in `logdir`.
        The other arguments to the constructor control the asynchronous writes to
        the event file.

        :param logdir: Directory where event file will be written.
        :type logdir:  str
        :param max_queue: Size of the queue for pending events and
            summaries before one of the 'add' calls forces a flush to disk.
        :type max_queue: int
        :param filename_suffix: Suffix added to all event filenames in the logdir directory.
            More details on filename construction in 
            tensorboard.summary.writer.event_file_writer.EventFileWriter.
        :type filename_suffix: str
        """
        self.logdir = str(logdir)
        self.event_writer = EventFileWriter(self.logdir, max_queue, filename_suffix)

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self.logdir

    def add_event(self, event, step=None, walltime=None):
        """Adds an event to the event file.

        :param event: An `Event` protocol buffer.
        :param step: Optional global step value for training process to record with the event.
        :type step: Number
        :param walltime: Given time to override the default walltime.
        :type walltime: Optional, float
        """
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            # Make sure step is converted from numpy or other formats
            # since protobuf might not convert depending on version
            event.step = int(step)
        self.event_writer.add_event(event)

    def add_summary(self, summary, global_step=None, walltime=None):
        """Adds a `Summary` protocol buffer to the event file.

        This method wraps the provided summary in an `Event` protocol buffer
        and adds it to the event file.

        :param summary: A `Summary` protocol buffer.
        :param global_step: Optional global step value for training process to record with the summary.
        :type global_step: Number
        :param walltime: Given time to override the default walltime.
        :type walltime: Optional, float
        """
        event = event_pb2.Event(summary=summary)
        self.add_event(event, global_step, walltime)

    def add_graph(self, GraphDef_proto, walltime=None):
        """Adds a `GraphDef` protocol buffer to the event file.

        :param graph_profile: A GraphDef protocol buffer.
        :param walltime: Optional walltime to override default
                        (current) walltime (from time.time()) seconds after epoch.
        :type walltime: Optional, float
        """
        event = event_pb2.Event(graph_def=GraphDef_proto.SerializeToString())
        self.add_event(event, None, walltime)

    def add_run_metadata(self, run_metadata, tag, global_step=None, walltime=None):
        """Adds a metadata information for a single session.run() call.

        :param run_metadata: A `RunMetadata` protobuf object.
        :param tag: The tag name for this metadata.
        :type tag: string
        :param global_step: global step counter to record with the StepStats.
        :type global_step: int
        :param walltime: Given time to override the default walltime.
        :type walltime: Optional, float
        """
        tagged_metadata = event_pb2.TaggedRunMetadata(
            tag=tag, run_metadata=run_metadata.SerializeToString())

        event = event_pb2.Event(tagged_run_metadata=tagged_metadata)
        self.add_event(event, global_step, walltime)

    def flush(self):
        """Flushes the event file to disk.

        Call this method to make sure that all pending events have been written to disk.
        """
        self.event_writer.flush()

    def close(self):
        """Flushes the event file to disk and close the file.

           Call this method when you do not need the summary writer anymore.
        """
        self.event_writer.close()

    def reopen(self):
        """Reopens the EventFileWriter.

        Can be called after `close()` to add more events in the same directory.
        The events will go into a new events file.
        Does nothing if the EventFileWriter was not closed.
        """
        self.event_writer.reopen()
