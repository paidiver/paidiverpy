paidiverpy.dashboard.runner
===========================

.. py:module:: paidiverpy.dashboard.runner

.. autoapi-nested-parse::

   Subprocess entry point used by the dashboard.

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   paidiverpy.dashboard.runner.EventHandler
   paidiverpy.dashboard.runner.MemoryMonitor


Module Contents
---------------

.. py:class:: EventHandler(level=NOTSET)

   Bases: :py:obj:`logging.Handler`


   
   Forward regular Paidiverpy logs as newline-delimited dashboard events.
















   ..
       !! processed by numpydoc !!

   .. py:method:: emit(record: logging.LogRecord) -> None

      
      Do whatever it takes to actually log the specified logging record.

      This version is intended to be implemented by subclasses and so
      raises a NotImplementedError.















      ..
          !! processed by numpydoc !!


.. py:class:: MemoryMonitor

   
   Sample runner and Dask child-process memory without changing core Paidiverpy.
















   ..
       !! processed by numpydoc !!

   .. py:method:: rss_mb() -> float

      
      Return aggregate RSS for the runner and all living descendants.
















      ..
          !! processed by numpydoc !!


   .. py:method:: sample() -> float

      
      Record and return the latest aggregate memory reading.
















      ..
          !! processed by numpydoc !!


   .. py:method:: loop() -> None

      
      Emit periodic samples until pipeline processing finishes.
















      ..
          !! processed by numpydoc !!


   .. py:method:: dataset_mb() -> float | None

      
      Estimate retained xarray storage without computing lazy arrays.
















      ..
          !! processed by numpydoc !!


   .. py:method:: handle(event: paidiverpy.pipeline.RunEvent) -> None

      
      Forward core lifecycle events and attach step memory summaries.
















      ..
          !! processed by numpydoc !!


