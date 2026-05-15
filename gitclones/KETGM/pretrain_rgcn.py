"""
R-GCN Pre-training — Step 4 of the KETGM pipeline.

Pre-trains the R-GCN autoencoder with link prediction (Eq 3)
and reconstruction loss (Eq 5), then extracts node embeddings.
Paper Section III-B2.
"""
import os
import time
import torch
from tqdm.auto import trange

from config import Config
from models import RGCNAutoencoder, sample_negatives


def run(kg, device, config=None):
    """
    Pre-train R-GCN autoencoder and extract embeddings.

    Parameters
    ----------
    kg     : dict from knowledge_graph.run()
    device : 'cuda' or 'cpu'

    Returns
    -------
    dict with keys:
        rgcn_model, all_tc, all_tm_c, topic_embeds, topic_tc,
        edge_index, edge_type
    """
    if config is None:
        config = Config

    num_nodes = len(kg['node2id'])
    num_rels  = len(kg['rel2id'])

    # Ensure there are edges to train on
    assert len(kg['edges']) > 0, "KG has no edges — check ConceptNet + topic words"

    edges_t = torch.tensor(kg['edges'], dtype=torch.long)     # (E, 3) [src, rel, tgt]
    edge_index = torch.stack([edges_t[:, 0], edges_t[:, 2]]).to(device)  # (2, E)
    edge_type  = edges_t[:, 1].to(device)                     # (E,)
    pos_triplets = edges_t.to(device)

    print(f'R-GCN setup: {num_nodes:,} nodes · {num_rels} rels · {edges_t.size(0):,} edges')

    rgcn_model = RGCNAutoencoder(
        num_nodes=num_nodes, num_rels=num_rels,
        hidden_dim=config.RGCN_HIDDEN,
        num_layers=config.RGCN_NUM_LAYERS,
        num_bases=min(config.RGCN_NUM_BASES, num_rels),
    ).to(device)

    opt_rgcn = torch.optim.Adam(rgcn_model.parameters(), lr=config.RGCN_LR)
    print(f'R-GCN params: {sum(p.numel() for p in rgcn_model.parameters()):,}')

    # ── Training loop ─────────────────────────────────────────────────
    t0 = time.time()
    rgcn_model.train()
    for epoch in (pbar := trange(config.RGCN_EPOCHS, desc='R-GCN')):
        neg_tri = sample_negatives(pos_triplets, num_nodes).to(device)
        link_loss, recon_loss, tc, tm_c = rgcn_model(
            edge_index, edge_type, pos_triplets, neg_tri
        )
        loss = link_loss + config.MU * recon_loss
        opt_rgcn.zero_grad()
        loss.backward()
        opt_rgcn.step()
        if (epoch+1) % 100 == 0 or epoch == 0:
            pbar.set_postfix(link=f'{link_loss:.4f}', recon=f'{recon_loss:.4f}')

    elapsed = time.time() - t0
    print(f'\n✓ R-GCN pre-training done  ({elapsed:.1f}s)')
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(rgcn_model.state_dict(),
               os.path.join(config.CHECKPOINT_DIR, 'rgcn_pretrained.pt'))

    # ── Extract node & topic embeddings ───────────────────────────────
    rgcn_model.eval()
    with torch.no_grad():
        all_tc  = rgcn_model.encode(edge_index, edge_type)   # (N, 200)
        all_tm_c = rgcn_model.feat_map(all_tc)                # (N, 200)

    # Handle case where some/all topic words are not in KG
    if len(kg['topic_node_ids']) > 0:
        topic_ids_t = torch.tensor(kg['topic_node_ids'], dtype=torch.long,
                                    device=device)
        topic_embeds = all_tm_c[topic_ids_t]                   # (T, 200)
        topic_tc     = all_tc[topic_ids_t]                     # (T, 200) — raw tc for topics
    else:
        topic_embeds = torch.zeros(1, config.RGCN_HIDDEN, device=device)
        topic_tc     = torch.zeros(1, config.RGCN_HIDDEN, device=device)
        print('⚠  No topic words found in KG — using zero fallback')

    print(f'Node embeddings : {all_tm_c.shape}')
    print(f'Topic embeddings: {topic_embeds.shape}')

    return dict(
        rgcn_model=rgcn_model,
        all_tc=all_tc,
        all_tm_c=all_tm_c,
        topic_embeds=topic_embeds,
        topic_tc=topic_tc,
        edge_index=edge_index,
        edge_type=edge_type,
    )
