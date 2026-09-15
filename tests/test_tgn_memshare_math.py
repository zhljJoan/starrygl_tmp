"""CPU component comparison against actual native MemShare source bodies.

This is deliberately not a claim of paired full-model/distributed training.
AST loading isolates native names without importing its CUDA runtime at import
time. The tested class/method bodies are unchanged; optional accelerators are off.
"""
import ast
import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import starrygl as sg
from starrygl.model.layers.temporal import TemporalTransformerAttentionLayer


@pytest.fixture
def native(monkeypatch):
    dgl=pytest.importorskip("dgl")
    root=Path(os.environ.get("MEMSHARE_REFERENCE_ROOT",Path.home()/"MemShare-public/MemShare"))
    source=root/"starrygl/module"
    if not (source/"layers.py").exists():pytest.skip("Native MemShare checkout is not available")
    namespace=dict(torch=torch,np=np,math=math,os=os,dgl=dgl,
        triton_segment_attention=SimpleNamespace(enabled=lambda:False),
        triton_edge_projection=SimpleNamespace(enabled=lambda:False))
    monkeypatch.setenv("MEMSHARE_COMPACT_KV_PROJECTION","off")
    tree=ast.parse((source/"layers.py").read_text())
    selected=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in {"TimeEncode","TransfomerAttentionLayer"}]
    assert len(selected)==2
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source/"layers.py"),"exec"),namespace)
    tree=ast.parse((source/"memorys.py").read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="AsyncMemeoryUpdater")
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=="rnn_updater")
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source/"memorys.py"),"exec"),namespace)
    return SimpleNamespace(**namespace)


def test_tgn_attention_matches_native_outputs_and_gradients(native):
    torch.manual_seed(804)
    ref=native.TransfomerAttentionLayer(4,2,3,2,0.,0.,4)
    layer=TemporalTransformerAttentionLayer(node_dim=4,edge_dim=2,time_dim=3,
        num_heads=2,out_dim=4,dropout=0.,att_dropout=0.,score_scale=1.)
    weights=copy.deepcopy(ref.state_dict())
    # Native concatenates [aggregation, dst_h]; common SG layer uses [dst_h, aggregation].
    weights["w_out.weight"]=weights["w_out.weight"][:,torch.tensor([4,5,6,7,0,1,2,3])]
    layer.load_state_dict(weights)
    src=torch.tensor([2,3,2,1]);dst=torch.tensor([0,0,1,1])
    block=sg.GraphBlock(src_nodes=torch.arange(4),dst_nodes=torch.arange(2),edge_ids=torch.arange(4),
        format="coo",row=src,col=dst,num_src=4,num_dst=2)
    graph=native.dgl.create_block((src,dst),num_src_nodes=4,num_dst_nodes=2)
    x=torch.randn(4,4,requires_grad=True);edge=torch.randn(4,2,requires_grad=True)
    dt=torch.tensor([.2,1.3,.7,2.1],requires_grad=True)
    rx=x.detach().clone().requires_grad_();re=edge.detach().clone().requires_grad_();rt=dt.detach().clone().requires_grad_()
    graph.srcdata["h"]=rx;graph.edata["f"]=re;graph.edata["dt"]=rt
    expected=ref(graph);actual=layer(block,x,edge,dt)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=2e-6)
    coefficient=torch.randn_like(actual)
    (actual*coefficient).sum().backward();(expected*coefficient).sum().backward()
    for value,reference in ((x,rx),(edge,re),(dt,rt)):
        torch.testing.assert_close(value.grad,reference.grad,rtol=1e-4,atol=2e-6)
    for name,value in layer.named_parameters():
        reference=dict(ref.named_parameters())[name].grad
        if name=="w_out.weight":reference=reference[:,torch.tensor([4,5,6,7,0,1,2,3])]
        torch.testing.assert_close(value.grad,reference,rtol=1e-4,atol=2e-6)


def test_tgn_gru_uses_query_cutoff_like_native_and_preserves_commit_times(native):
    torch.manual_seed(818)
    model=sg.TGNModel(in_dim=2,hidden_dim=3,out_dim=3,time_dim=4,edge_dim=1)
    graph=sg.graph_block_from_coo(src=torch.tensor([0,1,2]),dst=torch.tensor([1,2,0]),
        edge_ids=torch.arange(3),num_nodes=3,format="coo")
    query=torch.tensor([5.,5.,10.]);graph.srcdata["ts"]=query
    graph.edata["f"]=torch.zeros(3,1)
    memory=torch.randn(3,3,requires_grad=True);mail=torch.randn(3,7,requires_grad=True)
    prev=torch.tensor([.5,1.,1.5]);mail_ts=torch.tensor([[1.],[2.],[3.]])
    events=sg.EventRows(src=torch.tensor([0]),dst=torch.tensor([1]),edge_ids=torch.tensor([0]),ts=torch.tensor([5.]))
    batch=sg.Batch(mode="event",graph=graph,blocks=((graph,),),features={"x":(torch.ones(3,2),),"pos_edge_feat":(torch.zeros(1,1),)},
        state={"node_memory":memory,"node_memory_ts":prev,"mailbox":mail,"mailbox_ts":mail_ts},targets={"events":events})
    time=native.TimeEncode(4);time.load_state_dict(model.memory.time_enc.state_dict())
    gru=torch.nn.GRUCell(11,3);gru.load_state_dict(model.memory.updater.state_dict())
    reference=SimpleNamespace(dim_time=4,time_enc=time,compact_gru_update=False,compact_memory_ops=False,
        memory_param={"memory_update":"gru"},ceil_updater=gru)
    rm=memory.detach().clone().requires_grad_();rmail=mail.detach().clone().requires_grad_()
    native_block=SimpleNamespace(srcdata={"ts":query,"mem_ts":prev,"mem_input":rmail,"mem":rm})
    expected=native.rnn_updater(reference,native_block)
    output=model.encode(batch)
    torch.testing.assert_close(output.state_embeddings,expected,rtol=1e-4,atol=2e-6)
    coefficient=torch.randn_like(expected)
    (output.state_embeddings*coefficient).sum().backward();(expected*coefficient).sum().backward()
    for value,ref in ((memory,rm),(mail,rmail)):
        torch.testing.assert_close(value.grad,ref.grad,rtol=1e-4,atol=2e-6)
    for current,other in ((model.memory.updater,gru),(model.memory.time_enc,time)):
        for name,value in current.named_parameters():
            torch.testing.assert_close(value.grad,dict(other.named_parameters())[name].grad,rtol=1e-4,atol=2e-6)
    delta=model.state_update(batch,output)
    assert torch.equal(delta.node_ids,torch.tensor([0,1]))
    assert torch.equal(delta.timestamps,torch.tensor([5.,5.]))
    assert torch.equal(delta.metadata["mailbox_timestamps"],torch.tensor([5.,5.]))


def test_tgn_explicit_unscaled_attention_keeps_other_model_default():
    tgn=sg.TGNModel(in_dim=2,hidden_dim=4,out_dim=4,time_dim=3,num_heads=2)
    assert tgn.layers[0].score_scale==1.
    layer=TemporalTransformerAttentionLayer(node_dim=4,edge_dim=2,time_dim=3,
        num_heads=2,out_dim=4,dropout=0.,att_dropout=0.)
    assert layer.score_scale is None
