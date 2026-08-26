add_dataflow_scheduler_library(DataflowSchedulerKTIRToScheduleIR
    ComputeGroupExtraction.cpp
    ConstructThreeStagePipeline.cpp
    IndirectComputeGroupSplit.cpp
    KTIRLegalityCheck.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR

  GROUP CONVERSION
  EXPORT_NAME KTIRToScheduleIR

  DEPENDS
    DataflowSchedulerKTIRToScheduleIRPassesIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    KTIR::SpyreOpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ArchViews
    DataflowScheduler::KTDFAnalysis
    DataflowScheduler::KTDPLoweringDialect
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowScheduler::KTDFTransforms
    DataflowScheduler::KTDFUtils
    DataflowScheduler::TransformUtils
    DataflowScheduler::Utils

  LINK_MLIR_LIBS
    MLIRIR
    MLIRAffineDialect
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTransformUtils
)
